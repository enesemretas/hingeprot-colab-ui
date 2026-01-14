#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a HingeProt-like rigid-part summary from:
  - PDB_ID.hinge  (or hinges): hinge residues + chain
  - PDB_ID.new    : residue order (to know start/end and real ordering)

Goal (updated logic):
- Short fragments must NOT create new rigid-part boundaries.
- If a hinge would create a segment shorter than MIN_LEN, we treat that hinge as NON-EXISTENT
  (no sign change, no split). The would-be short segment is reported as a "Short Flexible Fragment",
  but it is fully included inside the final rigid part.

Output:
  PDB_ID.rigidparts.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------- parsers ----------------------
def _parse_pdb_like_chain_resid(line: str) -> Optional[Tuple[str, str]]:
    """
    Try to parse chain and resid from a PDB-like line.
    Supports both whitespace-split and fixed-column fallback.
    Returns (chain, resid_string) or None.
    """
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None

    parts = line.split()
    chain = ""
    resid = ""

    # Common case: "ATOM  1  CA  ALA A  12  ..."
    if len(parts) >= 6:
        chain = parts[4].strip()
        resid = parts[5].strip()
    else:
        # Fallback to fixed columns (PDB):
        # chain: col 22 (index 21), resid: cols 23-26 (22:26), icode: col 27 (index 26)
        if len(line) < 27:
            return None
        chain = line[21].strip()
        resnum = line[22:26].strip()
        icode = line[26].strip()
        resid = (resnum + icode).strip()

    if not chain or not resid:
        return None
    return chain, resid


def read_residue_order_from_new(new_path: Path) -> Dict[str, List[str]]:
    """
    Reads a PDB-like .new file and returns ordered residue IDs per chain.
    Residue ID is kept as a string (supports insertion codes like 269A).
    Deduplicates consecutive repeats (multiple atoms per residue).
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
      <idx> <resid> <chain>
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

            # More robust: take last two tokens as resid and chain
            resid_token = parts[-2].strip()
            chain = parts[-1].strip()

            if not resid_token or not chain:
                continue

            modes.setdefault(mode, {}).setdefault(chain, []).append(resid_token)

    return modes


# ---------------------- core logic ----------------------
def _merge_ranges(ranges: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """
    Merge overlapping/adjacent index ranges and clamp to [0, n-1].
    """
    if not ranges:
        return []

    # normalize + sort
    rr = []
    for a, b in ranges:
        if a > b:
            a, b = b, a
        rr.append((a, b))
    rr.sort()

    merged: List[Tuple[int, int]] = []
    for a, b in rr:
        a = max(0, a)
        b = min(n - 1, b)
        if a > b:
            continue
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a <= pb + 1:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))
    return merged


def select_kept_hinges_and_short_fragments(
    residue_list: List[str],
    hinge_resids: List[str],
    min_len: int,
) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
    """
    Key behavior:
    - Hinges that would create a segment shorter than min_len are treated as NON-EXISTENT.
    - Rigid parts will be built only from the kept hinges.
    - Short fragments are reported but do NOT split rigid parts.

    Returns:
      kept_hinges (resid strings)
      short_fragments as index ranges (start_idx, end_idx) in residue_list
      discarded_hinges (resid strings)  [informational]
    """
    n = len(residue_list)
    if n == 0:
        return [], [], []

    idx_map = {rid: i for i, rid in enumerate(residue_list)}

    # map hinge residues to indices that exist; keep in increasing order by sequence index
    hinge_idxs = []
    for r in hinge_resids:
        if r in idx_map:
            hinge_idxs.append(idx_map[r])
    hinge_idxs = sorted(set(hinge_idxs))

    kept_idxs: List[int] = []
    short_ranges: List[Tuple[int, int]] = []
    discarded_idxs: List[int] = []

    prev_kept = -1
    for idx in hinge_idxs:
        seg_len = idx - prev_kept  # residues count in (prev_kept+1 .. idx) inclusive
        if seg_len < min_len:
            # Treat this hinge as NON-EXISTENT (no boundary),
            # but record the would-be short segment (it will be included in final rigid part).
            short_ranges.append((prev_kept + 1, idx))
            discarded_idxs.append(idx)
        else:
            kept_idxs.append(idx)
            prev_kept = idx

    # Handle short tail: if last rigid part (after last kept hinge) would be too short,
    # drop that last hinge (so tail is merged into previous rigid part).
    # Repeat if necessary.
    while kept_idxs:
        last = kept_idxs[-1]
        tail_len = (n - 1) - last  # residues count in (last+1 .. end)
        if tail_len < min_len:
            short_ranges.append((last + 1, n - 1))
            discarded_idxs.append(last)
            kept_idxs.pop()
        else:
            break

    # Build resid outputs
    kept_hinges = [residue_list[i] for i in kept_idxs]
    discarded_hinges = [residue_list[i] for i in sorted(set(discarded_idxs)) if 0 <= i < n]

    short_ranges = _merge_ranges(short_ranges, n)
    return kept_hinges, short_ranges, discarded_hinges


def build_rigid_parts_from_kept_hinges(residue_list: List[str], kept_hinges: List[str]) -> List[Tuple[int, int]]:
    """
    Build rigid parts as index ranges using ONLY kept hinges.
    Hinge residue is included at the end of its left rigid part (classic convention):
      [start..hinge1], [hinge1+1..hinge2], ..., [last+1..end]
    """
    idx_map = {rid: i for i, rid in enumerate(residue_list)}
    kept_idxs = sorted({idx_map[r] for r in kept_hinges if r in idx_map})

    parts: List[Tuple[int, int]] = []
    start = 0
    for idx in kept_idxs:
        parts.append((start, idx))
        start = idx + 1

    if start <= len(residue_list) - 1:
        parts.append((start, len(residue_list) - 1))
    elif not parts and residue_list:
        parts.append((0, len(residue_list) - 1))

    return parts


def _fragment_part_no(fragment: Tuple[int, int], parts: List[Tuple[int, int]]) -> Optional[int]:
    a, b = fragment
    for i, (ps, pe) in enumerate(parts, start=1):
        if ps <= a and b <= pe:
            return i
    return None


# ---------------------- report ----------------------
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

            kept, short_frags, discarded = select_kept_hinges_and_short_fragments(
                residue_list=residue_list,
                hinge_resids=hinge_resids,
                min_len=min_len,
            )

            parts_idx = build_rigid_parts_from_kept_hinges(residue_list, kept)

            lines.append(f"Chain {chain}")
            lines.append("Rigid Part No\tResidues")
            for i, (ai, bi) in enumerate(parts_idx, start=1):
                lines.append(f"{i}\t\t{residue_list[ai]}-{residue_list[bi]}")

            if kept:
                lines.append("Hinge residues (kept): " + " ".join(kept))
            else:
                lines.append("Hinge residues (kept): (none)")

            if discarded:
                lines.append("Discarded hinges (treated as NON-hinge): " + " ".join(discarded))

            if short_frags:
                lines.append("")
                lines.append("Short Flexible Fragments (merged into rigid parts; no boundary created):")
                for i, (ai, bi) in enumerate(short_frags, start=1):
                    part_no = _fragment_part_no((ai, bi), parts_idx)
                    where = f" (in rigid part #{part_no})" if part_no is not None else ""
                    lines.append(f"{i}\t\t{residue_list[ai]}-{residue_list[bi]}{where}")

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
