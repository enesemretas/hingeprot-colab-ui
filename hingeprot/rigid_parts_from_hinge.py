#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a HingeProt-like rigid-part summary from:
  - PDB_ID.hinge  (or hinges): hinge residues + chain
  - PDB_ID.new    : residue order (to know start/end and real ordering)

Rule:
- If the segment length (in residue count) from:
    * start -> hinge
    * previous kept hinge -> hinge
  is < MIN_LEN, then:
    - do NOT keep that hinge
    - record that short segment as "Short Flexible Fragments"
    - include it inside the neighboring rigid part(s)

Output:
  PDB_ID.rigidparts.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_residue_order_from_new(new_path: Path) -> Dict[str, List[str]]:
    """
    Reads a PDB-like .new file and returns ordered residue IDs per chain.
    Residue ID is kept as a string (supports insertion codes like 269A).
    We deduplicate consecutive repeats (multiple atoms per residue).
    """
    residues_by_chain: Dict[str, List[str]] = {}
    last = None

    with new_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            chain = parts[4]
            resid = parts[5]  # may be "269" or "269A"

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
      <idx> <resid> <chain>
    Returns: modes[mode_number][chain] = [resid1, resid2, ...]
    """
    modes: Dict[int, Dict[str, List[str]]] = {}
    mode = None

    def _mode_from_header(s: str) -> int | None:
        s_low = s.lower()
        if "1st" in s_low:
            return 1
        if "2nd" in s_low:
            return 2
        # fallback: first integer in header
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

            # first column is row number / hinge row index (ignored)
            resid_token = parts[1]
            chain = parts[2]

            # keep resid as string; normalize (strip spaces)
            resid_token = resid_token.strip()

            modes.setdefault(mode, {}).setdefault(chain, []).append(resid_token)

    return modes


def filter_hinges_by_minlen(
    residue_list: List[str],
    hinge_resids: List[str],
    min_len: int,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Keeps hinge points that are at least min_len residues away from the previous kept hinge
    (start is treated as index -1).

    Returns:
      kept_hinges (resid strings)
      short_fragments as index ranges (start_idx, end_idx) in residue_list
    """
    idx_map = {rid: i for i, rid in enumerate(residue_list)}

    # only keep hinges that exist in residue_list
    hinge_idxs: List[int] = []
    hinge_ids: List[str] = []
    for r in hinge_resids:
        if r in idx_map:
            hinge_idxs.append(idx_map[r])
            hinge_ids.append(r)

    kept: List[str] = []
    short_frags: List[Tuple[int, int]] = []

    prev_kept_idx = -1
    for idx, rid in zip(hinge_idxs, hinge_ids):
        seg_len = idx - prev_kept_idx  # start->hinge is idx-(-1)=idx+1
        if seg_len < min_len:
            # short segment: prev_kept_idx+1 .. idx
            short_frags.append((prev_kept_idx + 1, idx))
            # do NOT update prev_kept_idx
        else:
            kept.append(rid)
            prev_kept_idx = idx

    # Optionally handle tiny tail as well (often useful in practice)
    if kept:
        last_idx = idx_map[kept[-1]]
        tail_len = (len(residue_list) - 1) - last_idx  # hinge+1 .. end
        if tail_len < min_len:
            short_frags.append((last_idx + 1, len(residue_list) - 1))
            kept = kept[:-1]

    # merge overlapping/adjacent short fragments
    short_frags = sorted(short_frags)
    merged: List[Tuple[int, int]] = []
    for a, b in short_frags:
        if a > b:
            a, b = b, a
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a <= pb + 1:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))

    # remove invalid
    merged = [(a, b) for a, b in merged if 0 <= a <= b < len(residue_list)]
    return kept, merged


def build_rigid_parts(residue_list: List[str], kept_hinges: List[str]) -> List[Tuple[str, str]]:
    """
    Rigid parts are contiguous segments split by kept hinges:
      [start..hinge1], [hinge1+1..hinge2], ..., [last+1..end]
    """
    idx_map = {rid: i for i, rid in enumerate(residue_list)}
    kept_idxs = sorted({idx_map[r] for r in kept_hinges if r in idx_map})

    parts: List[Tuple[str, str]] = []
    start = 0
    for idx in kept_idxs:
        parts.append((residue_list[start], residue_list[idx]))
        start = idx + 1
    if start <= len(residue_list) - 1:
        parts.append((residue_list[start], residue_list[-1]))
    return parts


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
                continue

            hinge_resids = modes[mode][chain]
            kept, short_frags = filter_hinges_by_minlen(residue_list, hinge_resids, min_len=min_len)
            parts = build_rigid_parts(residue_list, kept)

            lines.append(f"Chain {chain}")
            lines.append("Rigid Part No\tResidues")
            for i, (a, b) in enumerate(parts, start=1):
                lines.append(f"{i}\t\t{a}-{b}")

            if kept:
                lines.append("Hinge residues: " + " ".join(kept))
            else:
                lines.append("Hinge residues: (none)")

            if short_frags:
                lines.append("")
                lines.append("Short Flexible Fragments:")
                for i, (ai, bi) in enumerate(short_frags, start=1):
                    lines.append(f"{i}\t\t{residue_list[ai]}-{residue_list[bi]}")

            lines.append("")  # blank line between chains

        lines.append("")      # blank line between modes

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id", help="Example: 3LZG (expects 3LZG.hinge and 3LZG.new by default)")
    ap.add_argument("--hinge", default=None, help="Path to PDB_ID.hinge (or hinges). Default: <pdb_id>.hinge")
    ap.add_argument("--new", dest="newfile", default=None, help="Path to PDB_ID.new. Default: <pdb_id>.new")
    ap.add_argument("--min-len", type=int, default=15, help="Minimum segment length to accept a hinge (default=15)")
    ap.add_argument("--out", default=None, help="Output path. Default: <pdb_id>.rigidparts.txt")
    args = ap.parse_args()

    pdb_id = args.pdb_id
    hinge_path = Path(args.hinge) if args.hinge else Path(f"{pdb_id}.hinge")
    new_path = Path(args.newfile) if args.newfile else Path(f"{pdb_id}.new")
    out_path = Path(args.out) if args.out else Path(f"{pdb_id}.rigidparts.txt")

    residues_by_chain = read_residue_order_from_new(new_path)
    modes = parse_hinge_file(hinge_path)

    write_report(out_path, pdb_id, residues_by_chain, modes, min_len=args.min_len)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
