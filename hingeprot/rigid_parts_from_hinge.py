#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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


def parse_hinge_file(hinge_path: Path) -> Dict[int, Dict[str, List[str]]]:
    """
    Parses a .hinge/.hinges file with sections:
      ----> crosscorrelation : 1st slowest mode
      <idx> <resid> <chain>   (or extra cols; we take last two as resid & chain)
    Returns: modes[mode_number][chain] = [resid1, resid2, ...]
    """
    modes: Dict[int, Dict[str, List[str]]] = {}
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

            resid_token = parts[-2].strip()
            chain = parts[-1].strip()
            if not resid_token or not chain:
                continue

            modes.setdefault(mode, {}).setdefault(chain, []).append(resid_token)

    return modes


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


def build_parts_with_left_merge(
    residue_list: List[str],
    hinge_resids: List[str],
    min_len: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[str], List[str]]:
    """
    Core rule you asked:
    - Create initial parts from ALL hinge candidates.
    - If any part length < min_len => merge that whole part into the PREVIOUS part (left).
      (If it's the very first part, merge into the next because no previous exists.)

    Returns:
      parts_idx: final rigid parts as index ranges
      short_frags: the merged-away short parts as index ranges (merged for readability)
      kept_hinges: hinge residues that remain as boundaries (ends of parts except last)
      discarded_hinges: hinge residues that were removed
    """
    n = len(residue_list)
    if n == 0:
        return [], [], [], []

    idx_map = {rid: i for i, rid in enumerate(residue_list)}

    # map hinge residues -> indices that exist in residue_list
    hinge_idxs = sorted({idx_map[r] for r in hinge_resids if r in idx_map})

    # build initial parts using ALL hinges as boundaries (hinge belongs to left part end)
    parts: List[Tuple[int, int]] = []
    start = 0
    for h in hinge_idxs:
        if start <= h:
            parts.append((start, h))
            start = h + 1
    if start <= n - 1:
        parts.append((start, n - 1))
    if not parts:
        parts = [(0, n - 1)]

    def seg_len(seg: Tuple[int, int]) -> int:
        return seg[1] - seg[0] + 1

    short_raw: List[Tuple[int, int]] = []

    # merge short parts into PREVIOUS (left)
    i = 0
    while i < len(parts):
        if seg_len(parts[i]) < min_len and len(parts) > 1:
            short_raw.append(parts[i])

            if i == 0:
                # no previous: merge into next (forced)
                nxt = parts[1]
                parts[1] = (parts[0][0], nxt[1])
                parts.pop(0)
                i = 0
            else:
                # merge into previous
                prev = parts[i - 1]
                cur = parts[i]
                parts[i - 1] = (prev[0], cur[1])
                parts.pop(i)
                i = max(i - 1, 0)
        else:
            i += 1

    short_frags = _merge_ranges(short_raw, n)

    # kept hinges are the ends of parts except last part
    kept_idx_set = {end for (_, end) in parts[:-1]}
    kept_hinges = [residue_list[i] for i in sorted(kept_idx_set)]

    discarded_idx_set = set(hinge_idxs) - kept_idx_set
    discarded_hinges = [residue_list[i] for i in sorted(discarded_idx_set)]

    return parts, short_frags, kept_hinges, discarded_hinges


def write_report(
    out_path: Path,
    pdb_id: str,
    residues_by_chain: Dict[str, List[str]],
    modes: Dict[int, Dict[str, List[str]]],
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

            hinge_resids = modes[mode][chain]
            parts_idx, short_frags, kept, discarded = build_parts_with_left_merge(
                residue_list=residue_list,
                hinge_resids=hinge_resids,
                min_len=min_len,
            )

            lines.append(f"Chain {chain}")
            lines.append("Rigid Part No\tResidues")
            for i, (a, b) in enumerate(parts_idx, start=1):
                lines.append(f"{i}\t\t{residue_list[a]}-{residue_list[b]}")

            lines.append("Hinge residues (kept): " + (" ".join(kept) if kept else "(none)"))
            if discarded:
                lines.append("Discarded hinges (merged to previous rigid part): " + " ".join(discarded))

            if short_frags:
                lines.append("")
                lines.append("Short Flexible Fragments (fully merged into previous rigid part):")
                for i, (a, b) in enumerate(short_frags, start=1):
                    lines.append(f"{i}\t\t{residue_list[a]}-{residue_list[b]}")

            lines.append("")

        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id", help="Example: 3LZG (expects 3LZG.hinge and 3LZG.new by default)")
    ap.add_argument("--hinge", default=None, help="Path to PDB_ID.hinge (or hinges). Default: <pdb_id>.hinge")
    ap.add_argument("--new", dest="newfile", default=None, help="Path to PDB_ID.new. Default: <pdb_id>.new")
    ap.add_argument("--min-len", type=int, default=15, help="Minimum segment length (default=15)")
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
