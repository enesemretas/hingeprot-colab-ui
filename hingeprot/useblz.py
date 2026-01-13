#!/usr/bin/env python3
"""
useblz.py
---------

Python replacement for the original Fortran+BLZPACK 'useblz' stage.

Input  : upper-triangular sparse symmetric matrix file (e.g., "upperhessian")
Output : "<basename>.vwmatrixd" in Fortran list-directed style (spacing/wrapping like gfortran)

Key behavior:
- Computes k eigenpairs such that there are `target_pos` eigenvalues > 0.
  (k = target_pos + nneg, where nneg = count(eigenvalues < 0))
- Writes:
    1) header:   " nteig N"
    2) nteig lines:  eigenvalue  true_residual_norm
    3) eigenvectors flattened column-major: ((x(i,j), i=1..N), j=1..nteig)
- Formatting:
    * Leading space on every line (first token has gfortran sign-blank behavior)
    * 9 significant digits; scientific notation if |x| < 1e-4, else fixed
    * Between eigenvalue and residual: effectively 2 spaces when residual is positive
      (because of sign-blank in the token)
    * Eigenvector stream packed to max 79 chars/line (gfortran default feel)
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


# ------------------------ choose k for 36 positive ------------------------

def find_k_for_target_positive(A: np.ndarray, target_pos: int = 36) -> tuple[int, int]:
    """
    Returns (k, nneg) where:
      nneg = number of eigenvalues < 0
      k    = target_pos + nneg   so that among the smallest k eigenvalues,
             there will be target_pos positive ones (assuming negatives come first).
    """
    w = np.linalg.eigvalsh(A)
    nneg = int(np.sum(w < 0.0))
    k = int(target_pos + nneg)
    return k, nneg


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
    # leading space for the line + token(a) + separator + token(b)
    return " " + _token(a) + " " + _token(b) + "\n"

def _pack_stream(nums, max_width: int = 79):
    """
    Pack a stream of tokens into lines not exceeding max_width.
    Insert a single separator between tokens; the token's own sign-blank
    creates the "2 spaces for positive, 1 for negative" feel.
    """
    out_lines = []
    cur = ""
    for x in nums:
        t = _token(float(x))
        if not cur:
            cur = " " + t  # line-leading space
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
    """
    Writes:
      header
      eigenvalue/residual lines
      eigenvectors flattened column-major
    """
    nteig = int(eigvals.size)
    N = int(eigvecs.shape[0])

    with open(out_path, "w") as f:
        # Header: first char is a blank
        f.write(f" {nteig} {N}\n")

        for lam, res in zip(eigvals, residuals):
            f.write(_line_two_numbers(float(lam), float(res)))

        # ((x(i,j), i=1..N), j=1..nteig) => column-major flatten
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
    ap.add_argument("--target-pos", type=int, default=36, help="Number of eigenvalues > 0 desired (default: 36)")
    ap.add_argument("--max-width", type=int, default=79, help="Max line width for packed eigenvector stream (default: 79)")
    ap.add_argument("--out", default=None, help="Output filename (default: <base>.vwmatrixd)")
    args = ap.parse_args(argv)

    in_path = args.matrix
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Matrix file not found: {in_path}")

    A = read_upperhessian(in_path)

    k, nneg = find_k_for_target_positive(A, target_pos=args.target_pos)

    # Full dense solve (A is typically ~ 3Nres, still manageable for hingeprot sizes)
    w_all, V_all = np.linalg.eigh(A)
    w = w_all[:k]
    V = V_all[:, :k]

    # True residual norms: ||A v - λ v||_2
    R = A @ V - V * w
    res = np.linalg.norm(R, axis=0)

    out_path = args.out or _default_out_name(in_path)
    write_vwmatrix(out_path, w, res, V, max_width=args.max_width)

    # Minimal stdout for UI log
    pos_count = int(np.sum(w > 0.0))
    print(f"useblz.py: N={A.shape[0]}, nneg={nneg}, k={k}, positive_in_written={pos_count}")
    print(f"useblz.py: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
