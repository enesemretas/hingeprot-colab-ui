#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


# ------------------------- .new parsing -------------------------

def _parse_pdb_like_chain_resid(line: str) -> Optional[Tuple[str, str]]:
    """Try parse (chain, resid) from ATOM/HETATM lines (split or fixed-column fallback)."""
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None

    parts = line.split()
    if len(parts) >= 6:
        chain = parts[4].strip()
        resid = parts[5].strip()
        if chain and resid:
            return chain, resid

    # fixed-column fallback (PDB-ish)
    if len(line) < 27:
        return None
    chain = line[21].strip()
    resnum = line[22:26].strip()
    icode = line[26].strip()
    resid = (resnum + icode).strip()
    if chain and resid:
        return chain, resid
    return None


def read_residue_order_from_new(new_path: Path) -> Dict[str, List[str]]:
    """
    Reads a PDB-like .new file and returns ordered residue IDs per chain.
    Deduplicates consecutive repeats.
    """
    residues_by_chain: Dict[str, List[str]] = {}
    last: Optional[Tuple[str, str]] = None

    with new_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_pdb_like_chain_resid(line)
            if not parsed:
                continue
            chain, resid = parsed

            key = (chain, resid)
            if key == last:
                continue
            residues_by_chain.setdefault(chain, []).append(resid)
            last = key

    return residues_by_chain


# ------------------------- hinge parsing: keep seq_idx + resid + chain -------------------------

def parse_hinge_file(hinge_path: Path) -> Dict[int, Dict[str, List[Tuple[int, str]]]]:
    """
    Parses a .hinge/.hinges file with sections:
      ----> crosscorrelation : 1st slowest mode
      <seq_idx> <resid> <chain>   (or extra cols; we take first as seq_idx, last two as resid & chain)

    Returns:
      modes[mode_number][chain] = [(seq_idx, resid), ...]  (seq_idx is 1-based)
    """
    modes: Dict[int, Dict[str, List[Tuple[int, str]]]] = {}
    mode: Optional[int] = None

    def _mode_from_header(s: str) -> Optional[int]:
        s_low = s.lower()
        if "1st" in s_low:
            return 1
        if "2nd" in s_low:
            return 2
        m = re.search(r"(\d+)", s_low)
        return int(m.group(1)) if m else None

    with hinge_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("---->"):
                mode = _mode_from_header(line)
                continue

            if mode is None:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # seq_idx: first col
            try:
                seq_idx = int(float(parts[0]))
            except Exception:
                continue

            resid_token = parts[-2].strip()
            chain = parts[-1].strip()[:1]
            if not resid_token or not chain:
                continue

            modes.setdefault(mode, {}).setdefault(chain, []).append((seq_idx, resid_token))

    # ensure stable ordering
    for m in modes:
        for ch in modes[m]:
            modes[m][ch].sort(key=lambda x: x[0])

    return modes


# ------------------------- helpers -------------------------

def _merge_ranges(ranges: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    rr = []
    for a, b in ranges:
        if a > b:
            a, b = b, a
        a = max(0, a)
        b = min(n - 1, b)
        if a <= b:
            rr.append((a, b))
    rr.sort()

    merged: List[Tuple[int, int]] = []
    for a, b in rr:
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a <= pb + 1:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))
    return merged


# ------------------------- CORE: remove shortest-gap hinge pairs iteratively -------------------------

def build_parts_with_restart_pair_removal(
    residue_list: List[str],
    hinge_entries: List[Tuple[int, str]],  # [(seq_idx,resid), ...] sorted by seq_idx
    min_len: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[str], List[str], List[Tuple[str, str, int]]]:
    """
    İstenen algoritma:

    - Mesafe: hinge dosyasındaki satır numarası farkı (seq_idx farkı).
    - Short fragment: komşu iki hinge için (seq_idx[j] - seq_idx[i]) < min_len.
    - Seçim: tüm short çiftler arasından EN KÜÇÜK mesafeli çifti seç.
      (tie-break: daha küçük seq_idx ile başlayan)
    - Kaldırma: seçilen çiftin İKİ hinge'i de kaldırılır.
    - Restart: her kaldırmadan sonra tarama EN BAŞTAN yapılır.

    Not: boundary pozisyonu mapping önceliği:
      1) chain-local index = seq_idx - 1 (range içindeyse)
      2) değilse resid string -> residue_list index fallback

    Returns:
      parts_idx: final rigid parts as index ranges (contiguous) [0..h1], [h1+1..h2], ...
      short_frags: removed short segments as index ranges (h_left+1 .. h_right) merged for readability
      kept_hinges: kept hinge residues (resid tokens) in ascending boundary order
      discarded_hinges: discarded hinge residues (resid tokens) in removal order summary (sorted by position)
      removed_pairs: list of removed hinge pairs as (resid1,resid2, seq_gap)
    """
    n = len(residue_list)
    if n == 0:
        return [], [], [], [], []

    idx_map = {rid: i for i, rid in enumerate(residue_list)}

    # Build candidate hinge objects with:
    # - seq_idx (1-based from file)
    # - pos (0-based in residue_list; prefer seq_idx-1)
    # - resid token
    cand: List[Dict[str, Any]] = []
    for seq_idx, resid in hinge_entries:
        pos = seq_idx - 1
        if 0 <= pos < n:
            # trust seq_idx mapping (chain-local)
            cand.append({"seq": seq_idx, "pos": pos, "resid": resid})
        else:
            # fallback: resid string
            if resid in idx_map:
                cand.append({"seq": seq_idx, "pos": idx_map[resid], "resid": resid})

    # deduplicate by position (keep smallest seq if duplicates)
    tmp: Dict[int, Dict[str, Any]] = {}
    for h in cand:
        p = int(h["pos"])
        if p not in tmp or int(h["seq"]) < int(tmp[p]["seq"]):
            tmp[p] = h
    hinges = sorted(tmp.values(), key=lambda x: int(x["seq"]))

    removed_pairs: List[Tuple[str, str, int]] = []
    removed_frags: List[Tuple[int, int]] = []
    removed_pos: set[int] = set()

    def _recompute_and_pick_pair(hs: List[Dict[str, Any]]) -> Optional[int]:
        """Return index i for pair (i,i+1) to remove; choose minimal gap < min_len, tie by earliest."""
        best_i: Optional[int] = None
        best_gap: Optional[int] = None
        for i in range(len(hs) - 1):
            gap = int(hs[i + 1]["seq"]) - int(hs[i]["seq"])
            if gap < min_len:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_i = i
                elif gap == best_gap and best_i is not None:
                    # tie-break: earlier seq (already ordered, so keep earlier i)
                    pass
        return best_i

    # main iterative removal with restart
    while True:
        if len(hinges) < 2:
            break
        i = _recompute_and_pick_pair(hinges)
        if i is None:
            break

        h1 = hinges[i]
        h2 = hinges[i + 1]
        gap = int(h2["seq"]) - int(h1["seq"])

        removed_pairs.append((str(h1["resid"]), str(h2["resid"]), gap))
        removed_pos.add(int(h1["pos"]))
        removed_pos.add(int(h2["pos"]))

        # short fragment segment is between the two boundaries: (pos_left+1 .. pos_right)
        a = int(h1["pos"]) + 1
        b = int(h2["pos"])
        if a <= b:
            removed_frags.append((a, b))

        # remove BOTH hinges
        del hinges[i + 1]
        del hinges[i]
        # restart automatically by continuing loop

    # final kept boundary positions
    kept_pos = sorted({int(h["pos"]) for h in hinges if 0 <= int(h["pos"]) < n})
    kept_hinges = [residue_list[p] for p in kept_pos]

    discarded_hinges = [residue_list[p] for p in sorted(list(removed_pos)) if 0 <= p < n]

    # build rigid parts from kept boundaries (each boundary is end of left part)
    parts: List[Tuple[int, int]] = []
    start = 0
    for p in kept_pos:
        if start <= p:
            parts.append((start, p))
        start = p + 1
    if start <= n - 1:
        parts.append((start, n - 1))
    if not parts:
        parts = [(0, n - 1)]

    short_frags = _merge_ranges(removed_frags, n)
    return parts, short_frags, kept_hinges, discarded_hinges


# ------------------------- reporting -------------------------

def write_report(
    out_path: Path,
    pdb_id: str,
    residues_by_chain: Dict[str, List[str]],
    modes: Dict[int, Dict[str, List[Tuple[int, str]]]],
    min_len: int,
) -> None:
    lines: List[str] = []

    for mode in sorted(modes.keys()):
        lines.append(f"----> Slowest mode {mode}: {pdb_id}")

        for chain in sorted(modes[mode].keys()):
            residue_list = residues_by_chain.get(chain)
            if not residue_list:
                lines.append(f"(Chain {chain} not found in {pdb_id}.new; skipping)")
                lines.append("")
                continue

            hinge_entries = modes[mode][chain]  # [(seq_idx,resid),...]
            parts_idx, short_frags, kept, discarded = build_parts_with_restart_pair_removal(
                residue_list=residue_list,
                hinge_entries=hinge_entries,
                min_len=min_len,
            )

            lines.append(f"Chain {chain}")
            lines.append("Rigid Part No\tResidues")
            for i, (a, b) in enumerate(parts_idx, start=1):
                lines.append(f"{i}\t\t{residue_list[a]}-{residue_list[b]}")

            lines.append("Hinge residues (kept): " + (" ".join(kept) if kept else "(none)"))
            if discarded:
                lines.append("Discarded hinge residues (removed by short-pair rule): " + " ".join(discarded))

            if removed_pairs:
                lines.append("")
                lines.append("Removed short hinge-pairs (picked by smallest seq_idx gap; restart after each removal):")
                for k, (r1, r2, gap) in enumerate(removed_pairs, start=1):
                    lines.append(f"{k}\t\t{r1}-{r2}\t(seq_gap={gap})")

            if short_frags:
                lines.append("")
                lines.append("Short fragments (between removed hinge-pairs; treated as if sign never flipped):")
                for k, (a, b) in enumerate(short_frags, start=1):
                    lines.append(f"{k}\t\t{residue_list[a]}-{residue_list[b]}")

            lines.append("")

        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id", help="Example: 1v4s (expects 1v4s.hinge and 1v4s.new by default)")
    ap.add_argument("--hinge", default=None, help="Path to PDB_ID.hinge (or hinges). Default: <pdb_id>.hinge")
    ap.add_argument("--new", dest="newfile", default=None, help="Path to PDB_ID.new. Default: <pdb_id>.new")
    ap.add_argument("--min-len", type=int, default=15, help="Minimum seq_idx-gap length; shorter => remove hinge pair (default=15)")
    ap.add_argument("--out", default=None, help="Output path. Default: <pdb_id>.rigidparts.txt")
    args = ap.parse_args()

    pdb_id = args.pdb_id
    hinge_path = Path(args.hinge) if args.hinge else Path(f"{pdb_id}.hinge")
    new_path = Path(args.newfile) if args.newfile else Path(f"{pdb_id}.new")
    out_path = Path(args.out) if args.out else Path(f"{pdb_id}.rigidparts.txt")

    residues_by_chain = read_residue_order_from_new(new_path)
    if not residues_by_chain:
        raise RuntimeError(f"No residues parsed from {new_path} (is it PDB-like ATOM/HETATM format?)")

    modes = parse_hinge_file(hinge_path)
    if not modes:
        raise RuntimeError(f"No modes/hinges parsed from {hinge_path}")

    write_report(out_path, pdb_id, residues_by_chain, modes, min_len=args.min_len)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
