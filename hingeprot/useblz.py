#!/usr/bin/env python3
"""
useblz.py
---------

Python replacement for the original Fortran+BLZPACK 'useblz' stage.

Key behavior (UPDATED):
- Computes k eigenpairs such that there are `target_pos` eigenvalues > pos_thresh.
  (k = target_pos + nbad, where nbad = count(eigenvalues <= pos_thresh))
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np


# ------------------------ IO: read upperhessian ------------------------

def read_upperhessian(path: str) -> np.ndarray:
    """
    Read symmetric sparse upper-triangular coordinate list:
      first line: NA
      next lines: i j value   (1-based indices), only i<=j present

    Returns dense symmetric matrix A.
    """
    with open(path, "r") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"Empty matrix file: {path}")
        na = int(first.split()[0])  # not strictly required, but validates format

        rows, cols, vals = [], [], []
        maxij = 0
        for line in f:
            if not line.strip():
                continue
            i, j, v = line.split()[:3]
            i = int(i); j = int(j)
            v = float(v.replace("D", "E"))
            rows.append(i - 1)
            cols.append(j - 1)
            vals.append(v)
            if i > maxij: maxij = i
            if j > maxij: maxij = j

    N = maxij
    A = np.zeros((N, N), dtype=float)

    for r, c, v in zip(rows, cols, vals):
        A[r, c] = v
        if r != c:
            A[c, r] = v
    return A


# ------------------------ choose k for 36 eigenvalues > threshold ------------------------

def find_k_for_target_positive(A: np.ndarray, target_pos: int = 36, pos_thresh: float = 1e-5) -> tuple[int, int]:
    """
    Returns (k, nbad) where:
      nbad = number of eigenvalues <= pos_thresh
      k    = target_pos + nbad
    so that among the smallest k eigenvalues, there will be target_pos eigenvalues > pos_thresh (if available).
    """
    w = np.linalg.eigvalsh(A)  # sorted ascending
    nbad = int(np.sum(w <= pos_thresh))
    k = int(target_pos + nbad)
    # clamp to N (avoid out of range if matrix doesn't have enough > thresh eigenvalues)
    k = min(k, A.shape[0])
    return k, nbad


# ------------------------ Fortran-like formatting helpers ------------------------

def _fmt_9sig(x: float) -> str:
    """9 significant digits; use scientific notation if |x| < 1e-4 else fixed."""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax < 1e-4:
        return f"{x:.8E}"  # 1 digit + 8 decimals => 9 sig figs
    e = int(math.floor(math.log10(ax)))
    decimals = max(0, 9 - (e + 1))
    s = f"{x:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def _token(x: float) -> str:
    """
    gfortran list-directed "sign blank":
      + numbers start with a blank
      - numbers start with '-'
    """
    s = _fmt_9sig(float(x))
    return (" " + s) if x >= 0 else s

def _line_two_numbers(a: float, b: float) -> str:
    return " " + _token(a) + " " + _token(b) + "\n"

def _pack_stream(nums, max_width: int = 79):
    out_lines = []
    cur = ""
    for x in nums:
        t = _token(float(x))
        if not cur:
            cur = " " + t
        else:
            cand = cur + " " + t
            if len(cand) <= max_width:
                cur = cand
            else:
                out_lines.append(cur + "\n")
                cur = " " + t
    if cur:
        out_lines.append(cur + "\n")
    return out_lines


# ------------------------ write .vwmatrixd ------------------------

def write_vwmatrix(out_path: str, eigvals: np.ndarray, residuals: np.ndarray, eigvecs: np.ndarray, max_width: int = 79):
    nteig = int(eigvals.size)
    N = int(eigvecs.shape[0])

    with open(out_path, "w") as f:
        f.write(f" {nteig} {N}\n")

        for lam, res in zip(eigvals, residuals):
            f.write(_line_two_numbers(float(lam), float(res)))

        stream = np.asarray(eigvecs, dtype=float).ravel(order="F")
        f.writelines(_pack_stream(stream, max_width=max_width))


# ------------------------ main ------------------------

def _default_out_name(in_path: str) -> str:
    base = os.path.basename(in_path)
    if "." in base:
        base = base[: base.rfind(".")]
    return base + ".vwmatrixd"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compute eigenpairs from upperhessian and write <base>.vwmatrixd.")
    ap.add_argument("matrix", help="Input upper-triangular matrix file (e.g., upperhessian)")
    ap.add_argument("--target-pos", type=int, default=36, help="Number of eigenvalues > threshold desired (default: 36)")
    ap.add_argument("--pos-thresh", type=float, default=1e-5, help="Positivity threshold (default: 1e-5)")
    ap.add_argument("--max-width", type=int, default=79, help="Max line width for packed eigenvector stream (default: 79)")
    ap.add_argument("--out", default=None, help="Output filename (default: <base>.vwmatrixd)")
    args = ap.parse_args(argv)

    in_path = args.matrix
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Matrix file not found: {in_path}")

    A = read_upperhessian(in_path)

    k, nbad = find_k_for_target_positive(A, target_pos=args.target_pos, pos_thresh=args.pos_thresh)

    w_all, V_all = np.linalg.eigh(A)
    w = w_all[:k]
    V = V_all[:, :k]

    R = A @ V - V * w
    res = np.linalg.norm(R, axis=0)

    out_path = args.out or _default_out_name(in_path)
    write_vwmatrix(out_path, w, res, V, max_width=args.max_width)

    above = int(np.sum(w > args.pos_thresh))
    print(f"useblz.py: N={A.shape[0]}, n_le_thresh={nbad}, k={k}, above_thresh_in_written={above}, thresh={args.pos_thresh:g}")
    print(f"useblz.py: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
