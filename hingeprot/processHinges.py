from __future__ import annotations
import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np


# ------------------------- PDB parsing / formatting -------------------------

@dataclass(frozen=True)
class ResKey:
    chain: str
    resnum: int
    icode: str  # insertion code (single char or space)


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def read_pdb_atoms(pdb_path: str):
    """
    Returns:
      lines: list[str] original lines (without trailing newline)
      atom_idx: list[int] indices into `lines` for ATOM/HETATM records
      atom_keys: list[ResKey] per-atom residue key
      atom_xyz: np.ndarray shape (n_atoms,3)
      ca_keys: list[ResKey] residue keys for CA atoms in order of appearance
      ca_xyz: np.ndarray shape (n_res,3) CA coords in same order as ca_keys
    """
    lines: List[str] = []
    atom_idx: List[int] = []
    atom_keys: List[ResKey] = []
    atom_xyz: List[Tuple[float, float, float]] = []

    ca_keys: List[ResKey] = []
    ca_xyz: List[Tuple[float, float, float]] = []
    seen_ca: set[ResKey] = set()

    with open(pdb_path, "r", errors="ignore") as f:
        for li, raw in enumerate(f):
            line = raw.rstrip("\n")
            lines.append(line)

            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue

            # PDB fixed columns
            chain = (line[21] if len(line) > 21 else " ").strip() or "A"
            resnum = int(line[22:26])
            icode = (line[26] if len(line) > 26 else " ")
            key = ResKey(chain=chain, resnum=resnum, icode=icode)

            x = _safe_float(line[30:38])
            y = _safe_float(line[38:46])
            z = _safe_float(line[46:54])

            atom_idx.append(li)
            atom_keys.append(key)
            atom_xyz.append((x, y, z))

            atname = line[12:16].strip() if len(line) >= 16 else ""
            if atname == "CA" and key not in seen_ca:
                seen_ca.add(key)
                ca_keys.append(key)
                ca_xyz.append((x, y, z))

    if not ca_keys:
        raise ValueError("No CA atoms found in the input PDB. (aaMol.size() == 0 equivalent)")

    return (
        lines,
        atom_idx,
        atom_keys,
        np.asarray(atom_xyz, dtype=float),
        ca_keys,
        np.asarray(ca_xyz, dtype=float),
    )


def format_pdb_line_with_xyz_b(line: str, x: float, y: float, z: float, bfac: float) -> str:
    """
    Update x,y,z and B-factor in an ATOM/HETATM line while preserving the rest.
    """
    # Ensure line has at least 66 columns
    if len(line) < 66:
        line = line.ljust(66)

    head = line[:30]
    occ = line[54:60]  # keep occupancy as-is
    tail = line[66:] if len(line) > 66 else ""

    return f"{head}{x:8.3f}{y:8.3f}{z:8.3f}{occ}{bfac:6.2f}{tail}".rstrip()


# ------------------------- Vector + hinge parsing -------------------------

def read_vector_file(vec_path: str, n_res: int) -> np.ndarray:
    """
    Vector file format observed:
      idx  dx  dy  dz  (optional magnitude column)
    idx is 1-based and follows the CA residue order.
    """
    v = np.zeros((n_res, 3), dtype=float)
    with open(vec_path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                i = int(parts[0]) - 1
            except Exception:
                continue
            if 0 <= i < n_res:
                v[i, 0] = float(parts[1])
                v[i, 1] = float(parts[2])
                v[i, 2] = float(parts[3])
    return v


def parse_hinge_file(hinge_path: str) -> Dict[int, List[Tuple[int, int, str]]]:
    """
    Hinge file observed to have sections:
      "1st slowest mode"
      <pairs>
      "2nd slowest mode"
      <pairs>

    Each pair line looks like:
      <i>  <j>  <Chain>
    Example: 47  57  A
    """
    out: Dict[int, List[Tuple[int, int, str]]] = {1: [], 2: []}
    mode: Optional[int] = None

    with open(hinge_path, "r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if "1st slowest mode" in s:
                mode = 1
                continue
            if "2nd slowest mode" in s:
                mode = 2
                continue
            if s.startswith("---->"):
                continue
            if mode is None:
                continue

            parts = s.split()
            if len(parts) < 3:
                continue

            # tolerate accidental extra columns
            try:
                a = int(parts[0])
                b = int(parts[1])
                ch = parts[2].strip()[:1] or "A"
            except Exception:
                continue

            out[mode].append((a, b, ch))

    # Basic sanity
    if not out[1] and not out[2]:
        raise ValueError("No hinge pairs parsed from hinge file.")

    return out


# ------------------------- Heuristics to match 3LZG outputs -------------------------

def _centroid_for_range(
    ca_keys: List[ResKey],
    ca_xyz: np.ndarray,
    chain: str,
    lo: int,
    hi: int,
) -> np.ndarray:
    idx = [
        i for i, k in enumerate(ca_keys)
        if k.chain == chain and lo <= k.resnum <= hi
    ]
    if not idx:
        return np.zeros(3, dtype=float)
    return ca_xyz[idx].mean(axis=0)


def _mean_mag_for_range(
    ca_keys: List[ResKey],
    vec_mag: np.ndarray,
    chain: str,
    lo: int,
    hi: int,
) -> float:
    idx = [
        i for i, k in enumerate(ca_keys)
        if k.chain == chain and lo <= k.resnum <= hi
    ]
    if not idx:
        return float("inf")
    return float(np.mean(vec_mag[idx]))


def select_boundary_pairs(
    pairs: List[Tuple[int, int, str]],
    ca_keys: List[ResKey],
    ca_xyz: np.ndarray,
    vec_mag: np.ndarray,
    outlier_mult: float = 5.0,
) -> Tuple[int, int, List[int]]:
    """
    Selects (left_pair_idx, right_pair_idx, kept_indices) using heuristics that
    reproduce core intervals seen in your 3LZG outputs.

    Steps:
      - compute mean vector magnitude over each (a..b) range
      - drop outliers: mean > median * outlier_mult
      - pick right pair as the one with largest max(a,b)
      - pick left pair as (among kept, excluding right) the one whose centroid is farthest
        from the right pair centroid
    """
    if not pairs:
        return 0, 0, []

    mags = []
    cents = []
    for (a, b, ch) in pairs:
        lo, hi = (a, b) if a <= b else (b, a)
        mags.append(_mean_mag_for_range(ca_keys, vec_mag, ch, lo, hi))
        cents.append(_centroid_for_range(ca_keys, ca_xyz, ch, lo, hi))

    finite = [m for m in mags if math.isfinite(m)]
    med = float(np.median(finite)) if finite else 0.0

    kept = [
        i for i, m in enumerate(mags)
        if math.isfinite(m) and (med == 0.0 or m <= med * outlier_mult)
    ]
    if not kept:
        kept = list(range(len(pairs)))

    right = max(kept, key=lambda i: max(pairs[i][0], pairs[i][1]))
    right_c = cents[right]
    left_cands = [i for i in kept if i != right]
    left = right if not left_cands else max(left_cands, key=lambda i: float(np.linalg.norm(cents[i] - right_c)))

    return left, right, kept


def core_interval_from_pairs(pairs: List[Tuple[int, int, str]], left_idx: int, right_idx: int) -> Tuple[int, int, str]:
    """
    Core interval = [left_end+1, right_end] on the selected chain.
    """
    aL, bL, chL = pairs[left_idx]
    aR, bR, chR = pairs[right_idx]
    ch = chR  # in practice same chain
    left_end = max(aL, bL)
    right_end = max(aR, bR)
    if left_end > right_end:
        left_end, right_end = right_end, left_end
    return left_end + 1, right_end, ch


def compute_loops(
    pairs: List[Tuple[int, int, str]],
    exclude_idxs: set[int],
    loop_thr: int,
) -> List[Tuple[int, int, str]]:
    """
    Reproduces the 3LZG .loops behavior:

      (A) "sliding" pairs:
          consecutive pairs where (a2==a1+1 and b2==b1+1) -> loop [b1,b2]
      (B) "disjoint-shift" pairs:
          for each pair i, find the nearest later pair j such that a_j >= b_i (non-overlapping),
          and if (a_j-a_i) == (b_j-b_i) and 5<=shift<=loop_thr -> loop [b_i,b_j]

    Boundary pairs (left/right) are excluded to avoid generating loops that cross the core boundary.
    """
    kept = [(i, p) for i, p in enumerate(pairs) if i not in exclude_idxs]
    kept.sort(key=lambda t: (t[1][2], t[1][0], t[1][1]))  # by chain, then a,b

    loops = set()

    # (A) sliding consecutive
    for (i1, p1), (i2, p2) in zip(kept, kept[1:]):
        a1, b1, ch1 = p1
        a2, b2, ch2 = p2
        if ch1 != ch2:
            continue
        if a2 == a1 + 1 and b2 == b1 + 1 and abs(b2 - b1) <= loop_thr:
            loops.add((min(b1, b2), max(b1, b2), ch1))

    # (B) disjoint-shift nearest non-overlapping
    for idx, (i, p1) in enumerate(kept):
        a1, b1, ch1 = p1
        for jdx in range(idx + 1, len(kept)):
            a2, b2, ch2 = kept[jdx][1]
            if ch2 != ch1:
                continue
            if a2 >= b1:
                shift_a = a2 - a1
                shift_b = b2 - b1
                if shift_a == shift_b and 5 <= shift_b <= loop_thr:
                    loops.add((min(b1, b2), max(b1, b2), ch1))
                break

    return sorted(loops, key=lambda t: (t[0], t[1], t[2]))


# ------------------------- Writing outputs -------------------------

def write_loops_file(out_path: str, loops_by_mode: Dict[int, List[Tuple[int, int, str]]]):
    with open(out_path, "w") as f:
        for mode in (1, 2):
            for lo, hi, ch in loops_by_mode.get(mode, []):
                f.write(f"{mode} {lo} {hi} {ch}:\n")


def write_moved_pdb(
    out_path: str,
    lines: List[str],
    atom_idx: List[int],
    atom_keys: List[ResKey],
    atom_xyz: np.ndarray,
    ca_keys: List[ResKey],
    vec: np.ndarray,  # per-CA-residue vectors
    core_start: int,
    core_end: int,
    core_chain: str,
    clustering_dist_thr: float,
):
    # Scale so that the maximum displacement from MODEL6 to MODEL11 is ~ clustering_dist_thr.
    # With factors -5..+5, MODEL11 uses factor +5, so per-step max ~ clustering_dist_thr/5.
    vec_mag = np.linalg.norm(vec, axis=1)
    max_mag = float(vec_mag.max()) if vec_mag.size else 1.0
    if max_mag <= 0:
        max_mag = 1.0
    per_step_target = float(clustering_dist_thr) / 5.0
    scale = per_step_target / max_mag

    # Map CA vectors to residue keys (one vector per residue)
    resvec: Dict[ResKey, np.ndarray] = {k: vec[i] for i, k in enumerate(ca_keys)}

    # Precompute B-factor mask by residue number (chain-aware)
    def bfac_for_key(k: ResKey) -> float:
        if k.chain == core_chain and (core_start <= k.resnum <= core_end):
            return 10.00
        return 5.00

    factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    with open(out_path, "w") as f:
        for mi, fac in enumerate(factors, start=1):
            f.write(f"MODEL{mi:9d}\n")
            # copy base lines (will patch ATOM/HETATM lines)
            out_lines = list(lines)

            # Apply per-residue displacement to every atom line
            for ai, li in enumerate(atom_idx):
                k = atom_keys[ai]
                disp = resvec.get(k, None)
                if disp is None:
                    continue
                dx, dy, dz = (disp * scale * float(fac))
                x0, y0, z0 = atom_xyz[ai]
                x, y, z = (x0 + dx, y0 + dy, z0 + dz)
                out_lines[li] = format_pdb_line_with_xyz_b(out_lines[li], x, y, z, bfac_for_key(k))

            # Write out, skipping any existing MODEL/ENDMDL in the input
            for line in out_lines:
                if line.startswith("MODEL") or line.startswith("ENDMDL"):
                    continue
                f.write(line.rstrip() + "\n")

            f.write("ENDMDL\n")


# ------------------------- CLI -------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("pdb_file")
    ap.add_argument("hinge_file")
    ap.add_argument("vector_file1")
    ap.add_argument("vector_file2")
    ap.add_argument("loop_thr", type=int)
    ap.add_argument("clustering_dist_thr", type=float)
    args = ap.parse_args(argv)

    (
        lines,
        atom_idx,
        atom_keys,
        atom_xyz,
        ca_keys,
        ca_xyz,
    ) = read_pdb_atoms(args.pdb_file)

    hinge = parse_hinge_file(args.hinge_file)

    v1 = read_vector_file(args.vector_file1, n_res=len(ca_keys))
    v2 = read_vector_file(args.vector_file2, n_res=len(ca_keys))

    # Select boundaries + core intervals
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)

    left1, right1, kept1 = select_boundary_pairs(hinge.get(1, []), ca_keys, ca_xyz, mag1, outlier_mult=5.0)
    left2, right2, kept2 = select_boundary_pairs(hinge.get(2, []), ca_keys, ca_xyz, mag2, outlier_mult=5.0)

    core1_start, core1_end, core1_chain = core_interval_from_pairs(hinge[1], left1, right1)
    core2_start, core2_end, core2_chain = core_interval_from_pairs(hinge[2], left2, right2)

    # Loops (exclude boundary pairs)
    loops1 = compute_loops(hinge.get(1, []), exclude_idxs={left1, right1}, loop_thr=args.loop_thr)
    loops2 = compute_loops(hinge.get(2, []), exclude_idxs={left2, right2}, loop_thr=args.loop_thr)

    # Write outputs next to pdb file
    out_loops = args.pdb_file + ".loops"
    out_m1 = args.pdb_file + ".moved1.pdb"
    out_m2 = args.pdb_file + ".moved2.pdb"

    write_loops_file(out_loops, {1: loops1, 2: loops2})

    write_moved_pdb(
        out_m1,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v1,
        core1_start, core1_end, core1_chain,
        args.clustering_dist_thr,
    )
    write_moved_pdb(
        out_m2,
        lines, atom_idx, atom_keys, atom_xyz, ca_keys,
        v2,
        core2_start, core2_end, core2_chain,
        args.clustering_dist_thr,
    )

    print(f"Wrote: {out_loops}")
    print(f"Wrote: {out_m1}")
    print(f"Wrote: {out_m2}")
    print(f"Mode1 core (bf=10): {core1_chain}:{core1_start}-{core1_end}")
    print(f"Mode2 core (bf=10): {core2_chain}:{core2_start}-{core2_end}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
