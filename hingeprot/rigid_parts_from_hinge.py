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
    Parses a .hinge file with sections:
      ----> crosscorrelation : 1st slowest mode
      <seq_idx> <resid> <chain>

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

            try:
                seq_idx = int(float(parts[0]))
            except Exception:
                continue

            resid_token = parts[-2].strip()
            chain = parts[-1].strip()[:1]
            if not resid_token or not chain:
                continue

            modes.setdefault(mode, {}).setdefault(chain, []).append((seq_idx, resid_token))

    # stable ordering by seq_idx
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
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    İstenen algoritma (raporda sadece rigid parts + short fragments göstereceğiz):

    - Mesafe: hinge dosyasındaki satır numarası farkı (seq_idx farkı).
    - Short fragment: komşu iki hinge için (seq_idx[j] - seq_idx[i]) < min_len.
    - Seçim: tüm short çiftler arasından EN KÜÇÜK seq_idx-gap çifti seç.
      (tie-break: daha küçük seq_idx ile başlayan)
    - Kaldırma: seçilen çiftin İKİ hinge'i de kaldırılır.
    - Restart: her kaldırmadan sonra tarama EN BAŞTAN yapılır.

    Boundary pozisyonu (chain içi index) bulma:
      - Önce resid token'ı residue_list içinde arar (daha güvenli).
      - Bulamazsa seq_idx-1 fallback (uygunsa).

    Returns:
      parts_idx: final rigid parts index ranges
      short_frags: removed short fragments as index ranges (between removed hinge-pairs) merged
    """
    n = len(residue_list)
    if n == 0:
        return [], []

    idx_map = {rid: i for i, rid in enumerate(residue_list)}

    hinges: List[Dict[str, Any]] = []
    for seq_idx, resid in hinge_entries:
        if resid in idx_map:
            pos = idx_map[resid]
        else:
            pos = seq_idx - 1
            if not (0 <= pos < n):
                continue
        hinges.append({"seq": int(seq_idx), "pos": int(pos), "resid": str(resid)})

    # deduplicate by position (keep smallest seq)
    tmp: Dict[int, Dict[str, Any]] = {}
    for h in hinges:
        p = int(h["pos"])
        if p not in tmp or int(h["seq"]) < int(tmp[p]["seq"]):
            tmp[p] = h

    hinges = sorted(tmp.values(), key=lambda x: int(x["seq"]))

    removed_frags: List[Tuple[int, int]] = []

    def _pick_pair_index(hs: List[Dict[str, Any]]) -> Optional[int]:
        best_i: Optional[int] = None
        best_gap: Optional[int] = None
        for i in range(len(hs) - 1):
            gap = int(hs[i + 1]["seq"]) - int(hs[i]["seq"])
            if gap < min_len:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_i = i
                # tie-break: earlier i already wins
        return best_i

    # iterative remove with restart
    while True:
        if len(hinges) < 2:
            break
        i = _pick_pair_index(hinges)
        if i is None:
            break

        h1 = hinges[i]
        h2 = hinges[i + 1]

        # short fragment is between boundaries: (pos_left+1 .. pos_right)
        a = int(h1["pos"]) + 1
        b = int(h2["pos"])
        if a <= b:
            removed_frags.append((a, b))

        # remove BOTH hinges
        del hinges[i + 1]
        del hinges[i]
        # restart automatically

    kept_pos = sorted({int(h["pos"]) for h in hinges if 0 <= int(h["pos"]) < n})

    # build rigid parts from kept boundaries
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
    return parts, short_frags


# ------------------------- reporting (ONLY rigid parts + short fragments) -------------------------

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

            hinge_entries = modes[mode][chain]
            parts_idx, short_frags = build_parts_with_restart_pair_removal(
                residue_list=residue_list,
                hinge_entries=hinge_entries,
                min_len=min_len,
            )

            lines.append(f"Chain {chain}")
            lines.append("Rigid Part No\tResidues")
            for i, (a, b) in enumerate(parts_idx, start=1):
                lines.append(f"{i}\t\t{residue_list[a]}-{residue_list[b]}")

            if short_frags:
                lines.append("")
                lines.append("Short fragments:")
                for k, (a, b) in enumerate(short_frags, start=1):
                    lines.append(f"{k}\t\t{residue_list[a]}-{residue_list[b]}")

            lines.append("")

        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id", help="Example: 1v4s (expects 1v4s.hinge and 1v4s.new by default)")
    ap.add_argument("--hinge", default=None, help="Path to PDB_ID.hinge. Default: <pdb_id>.hinge")
    ap.add_argument("--new", dest="newfile", default=None, help="Path to PDB_ID.new. Default: <pdb_id>.new")
    ap.add_argument("--min-len", type=int, default=15,
                    help="Minimum seq_idx-gap length; shorter => remove hinge pair (default=15)")
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
