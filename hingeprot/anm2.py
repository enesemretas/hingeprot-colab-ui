#!/usr/bin/env python3
"""
anm2.py — Python translation of your Fortran "rbp" program (ANM Hessian writer).

Inputs (in current working directory by default):
  - anmcutoff   : single float (rcanm)
  - alpha.cor   : PDB-like ATOM records (we extract CA coordinates)

Output:
  - upperhessian
      first line: number_of_nonzero_upper_triangle_entries
      then lines: i  j  value   (1-indexed) for dn(i,j) where j>=i and dn!=0

Formatting goal (to match your Fortran example):
  - 1 leading space at the start of EVERY line
  - between 2nd column and value:
      * 2 spaces if value is positive
      * 1 space if value is negative
  - numeric formatting uses .10G so E-notation appears for tiny values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def read_single_float(path: str | Path) -> float:
    txt = Path(path).read_text(encoding="utf-8").strip().split()
    if not txt:
        raise ValueError(f"Empty file: {path}")
    return float(txt[0])


def read_ca_coords_from_alpha_cor(path: str | Path) -> np.ndarray:
    """
    Robust parser for PDB-like ATOM lines.
    Extracts only CA coordinates.

    Tries fixed-column PDB parsing first; falls back to whitespace parsing.
    """
    coords: list[tuple[float, float, float]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            # PDB fixed columns attempt
            try:
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
                continue
            except Exception:
                pass

            # Fallback: whitespace split
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                atom_name = parts[2].strip()
                if atom_name != "CA":
                    continue
                x, y, z = map(float, parts[-3:])
                coords.append((x, y, z))
            except Exception:
                continue

    if not coords:
        raise RuntimeError(f"No CA atoms found in {path}. Is alpha.cor PDB-like?")
    return np.array(coords, dtype=np.float64)


def build_sparse_upper_hessian(
    coords: np.ndarray,
    rcut: float,
    eps2: float = 1e-12
) -> dict[tuple[int, int], float]:
    """
    Builds ANM Hessian (3N x 3N), sparse, storing only the GLOBAL upper triangle (i<=j).

    For each contact (i<j) with ||ri-rj|| <= rcut:
      B = outer(d, d) / r^2   (d = ri - rj)
      H_ii += B
      H_jj += B
      H_ij -= B
    """
    n = int(coords.shape[0])
    if n <= 0:
        return {}

    coords_centered = coords - coords.mean(axis=0, keepdims=True)
    rcut2 = float(rcut) * float(rcut)

    acc: dict[tuple[int, int], float] = {}

    def add_to_upper(i: int, j: int, val: float) -> None:
        # store only global upper triangle
        if i <= j:
            key = (i, j)
        else:
            key = (j, i)
        acc[key] = acc.get(key, 0.0) + float(val)

    def add_diag_block(base: int, B: np.ndarray, sign: float) -> None:
        # only upper triangle within 3x3 block
        for a in range(3):
            ia = base + a
            for b in range(a, 3):
                ib = base + b
                add_to_upper(ia, ib, sign * B[a, b])

    def add_off_block(base_i: int, base_j: int, B: np.ndarray, sign: float) -> None:
        # for i<j, all entries are in global upper
        for a in range(3):
            ia = base_i + a
            for b in range(3):
                ib = base_j + b
                add_to_upper(ia, ib, sign * B[a, b])

    for i in range(n - 1):
        diffs = coords_centered[i] - coords_centered[i + 1 :]
        r2 = np.einsum("ij,ij->i", diffs, diffs)
        mask = r2 <= rcut2
        if not np.any(mask):
            continue

        js = np.nonzero(mask)[0] + (i + 1)
        base_i = 3 * i

        for j in js:
            d = coords_centered[i] - coords_centered[j]
            rr2 = float(d @ d)
            if rr2 <= eps2:
                continue

            B = np.outer(d, d) / rr2  # ga=1.0
            base_j = 3 * j

            add_diag_block(base_i, B, +1.0)
            add_diag_block(base_j, B, +1.0)
            add_off_block(base_i, base_j, B, -1.0)

    return acc


def write_upperhessian(
    sparse_upper: dict[tuple[int, int], float],
    outpath: str | Path,
    tol: float = 0.0
) -> None:
    """
    Write in the spacing style you showed from Fortran:

      " 98365"
      " 1 1  1.53889489"     (positive => TWO spaces between col2 and value)
      " 1 2 -0.0136846798"   (negative => ONE space between col2 and value)

    Numeric formatting: .10G (so tiny values appear as ...E-06, etc.)
    """
    items = [((i, j), v) for (i, j), v in sparse_upper.items() if abs(v) > tol]
    items.sort(key=lambda t: (t[0][0], t[0][1]))

    out = Path(outpath)
    with out.open("w", encoding="utf-8") as f:
        # leading space on first line
        f.write(f" {len(items)}\n")

        for (i, j), v in items:
            vstr = f"{v:.10G}"  # Fortran-like: switches to E for small values
            sep = "  " if not vstr.startswith("-") else " "  # EXACT rule you requested
            # leading space at line start
            f.write(f" {i+1} {j+1}{sep}{vstr}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="alpha.cor", help="alpha.cor (PDB-like ATOM lines)")
    ap.add_argument("--cutoff", default="anmcutoff", help="anmcutoff file (single float)")
    ap.add_argument("--out", default="upperhessian", help="output file name")
    ap.add_argument("--tol", type=float, default=0.0, help="drop entries with abs(val) <= tol")
    args = ap.parse_args()

    rcanm = read_single_float(args.cutoff)
    coords = read_ca_coords_from_alpha_cor(args.alpha)

    sparse_upper = build_sparse_upper_hessian(coords, rcanm)
    write_upperhessian(sparse_upper, args.out, tol=args.tol)


if __name__ == "__main__":
    main()
