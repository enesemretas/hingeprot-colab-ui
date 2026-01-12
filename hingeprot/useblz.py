#!/usr/bin/env python3
"""
useblz.py
---------

SciPy-based replacement for BLZPACK/MA47 eigen solve for ANM.

Input: upper-triangular coordinate (triplet) file (Fortran-style 1-indexed):
  line 1:  NA            (declared number of stored triplets)
  next lines:  i  j  a_ij (1-indexed indices; usually only upper triangle)

Output (default): <input>.vwmatrix
  line 1:  (one leading space) "k n"
  next k lines:  eigenvalue  imag_part   (imag_part is written as 0.0)
  then:  eigenvectors, written in Fortran column-major order (all of v1, then v2, ...),
         5 numbers per line.

Defaults (as requested):
  - k     = 38
  - sigma = 2.22044605E-16 (machine epsilon for float64)

Formatting rules (per user request):
  1) Header line starts with exactly one space.
  2) Other lines start with one leading space if first number is negative,
     otherwise two leading spaces.
  3) Between numbers on the same line:
       - if the next number is negative -> 1 space before it
       - else -> 2 spaces before it
  4) Numbers use 9 significant figures in fixed format until 0.0001; below that -> scientific,
     with mantissa having 8 decimals and an 'E' exponent (e.g., -3.30213541E-07).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional, Tuple

import numpy as np

DEFAULT_K = 38
DEFAULT_SIGMA = 2.22044605e-16  # machine epsilon (float64)
DEFAULT_TOL = 0.0


def _ensure_scipy():
    try:
        import scipy  # noqa: F401
        return
    except Exception:
        import subprocess
        subprocess.run(["python3", "-m", "pip", "-q", "install", "numpy", "scipy"], check=False)


def read_upper_tri_coo(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Read a Fortran-style coordinate file containing (i,j,a) triplets (1-indexed).
    Returns:
      rows0, cols0, data0  (0-indexed arrays)
      n                   (matrix dimension, inferred from max index)
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # first non-empty non-comment line should contain NA
        first = ""
        while True:
            first = f.readline()
            if first == "":
                raise ValueError("Empty matrix file.")
            s = first.strip()
            if s and not s.startswith(("#", "!", "c", "C")):
                break

        try:
            na_declared = int(s.split()[0])
        except Exception as e:
            raise ValueError(f"Failed to parse first line as integer NA: {s!r}") from e

        nmax = 0
        na_read = 0

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] in "#!":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            i = int(parts[0])
            j = int(parts[1])
            a = float(parts[2])

            rows.append(i - 1)
            cols.append(j - 1)
            data.append(a)

            if i > nmax:
                nmax = i
            if j > nmax:
                nmax = j

            na_read += 1
            if na_read >= na_declared:
                break

        if na_read == 0 or nmax <= 0:
            raise ValueError("No valid (i,j,val) triplets were read (check file format).")

        if na_read < na_declared:
            # Some generators put a wrong NA in the header; keep going with what we have.
            print(f"Warning: header NA={na_declared} but only {na_read} triplets were present.")

    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
        int(nmax),
    )


def build_symmetric_sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int):
    """
    Build a symmetric sparse matrix from stored (possibly upper-triangular) triplets.
    Mirrors off-diagonal entries.
    """
    _ensure_scipy()
    from scipy.sparse import coo_matrix

    off = rows != cols
    rr = np.concatenate([rows, cols[off]])
    cc = np.concatenate([cols, rows[off]])
    dd = np.concatenate([data, data[off]])

    A = coo_matrix((dd, (rr, cc)), shape=(n, n)).tocsc()
    A.sum_duplicates()
    return A


def compute_eigs_shift_invert(A, k: int, sigma: float, tol: float = DEFAULT_TOL,
                             maxiter: int | None = None, ncv: int | None = None):
    """
    Compute k eigenpairs of symmetric sparse A near sigma using shift-invert.
    Returns evals (ascending), evecs (n x k).
    """
    _ensure_scipy()
    from scipy.sparse import eye
    from scipy.sparse.linalg import eigsh, splu, LinearOperator

    n = A.shape[0]
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0.")
    if k >= n:
        k = max(1, n - 1)

    sig = float(sigma)
    if sig == 0.0:
        raise ValueError("sigma must be non-zero for ANM Hessians (rigid-body zero modes make A singular).")

    # Factorize (A - sigma I) ONCE (constant sigma)
    Ashift = (A - sig * eye(n, format="csc", dtype=np.float64))
    try:
        lu = splu(Ashift)
    except Exception as e:
        raise RuntimeError(
            f"Shifted factorization failed for sigma={sig:.3e}. "
            f"If this happens due to numerical pivoting, try a slightly larger sigma (e.g., 1e-12). "
            f"Original error: {e}"
        ) from e

    OPinv = LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)

    evals, evecs = eigsh(
        A,
        k=k,
        sigma=sig,
        which="LM",
        OPinv=OPinv,
        tol=float(tol),
        maxiter=maxiter,
        ncv=ncv,
        return_eigenvectors=True,
    )

    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    return evals, evecs


def format_9sig(x: float) -> str:
    """
    Format number using the requested 9-significant-figure rules.

    - fixed until |x| < 1e-4 -> scientific (8 decimals in mantissa)
    - for 0 < |x| < 1: decimals = 9 + (# leading zeros after decimal)
    """
    if x == 0.0:
        return "0.000000000"

    ax = abs(x)

    # scientific for < 1e-4 (E-05 or smaller)
    if ax < 1e-4:
        return f"{x:.8E}"

    # fixed-point
    if ax >= 1.0:
        int_digits = int(math.floor(math.log10(ax))) + 1
        dec = max(0, 9 - int_digits)
        return f"{x:.{dec}f}"

    # 0 < ax < 1: count leading zeros after decimal
    z = int(math.floor(-math.log10(ax) - 1e-12))  # z=0 for [0.1,1), 1 for [0.01,0.1), etc.
    if z > 3:
        return f"{x:.8E}"
    dec = 9 + z
    return f"{x:.{dec}f}"


def format_line(nums: list[float], header: bool = False) -> str:
    """
    Apply spacing rules:
      - Header: exactly one leading space; integers separated by one space.
      - Other: each number is preceded by:
          * first number: 1 space if negative else 2 spaces
          * subsequent numbers: 1 space if negative else 2 spaces
    This yields:
      - row-leading space rule
      - 2 spaces between positive numbers, 1 before negatives
    """
    if header:
        return " " + " ".join(str(int(v)) for v in nums)

    out = []
    for v in nums:
        s = format_9sig(float(v))
        out.append((" " if float(v) < 0 else "  ") + s)
    return "".join(out)


def write_vwmatrix(out_path: str, evals: np.ndarray, evecs: np.ndarray):
    """
    Write BLZPACK-like .vwmatrix file with the spacing/significant-figure rules.
      line 1: " k n" (one leading space)
      next k lines: eigenvalue  0.0
      then eigenvectors in Fortran column-major order, 5 numbers per line.
    """
    n, k = evecs.shape
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(format_line([k, n], header=True) + "\n")

        for lam in evals:
            f.write(format_line([float(lam), 0.0]) + "\n")

        flat = np.asarray(evecs, dtype=np.float64).reshape(n * k, order="F")
        per_line = 5
        for i in range(0, flat.size, per_line):
            f.write(format_line(flat[i:i + per_line].tolist()) + "\n")


def solve_upperhessian_to_vwmatrix(matrix_file: str,
                                  k: int = DEFAULT_K,
                                  sigma: float = DEFAULT_SIGMA,
                                  out: str | None = None,
                                  tol: float = DEFAULT_TOL,
                                  maxiter: int | None = None,
                                  ncv: int | None = None) -> str:
    t0 = time.time()
    rows, cols, data, n = read_upper_tri_coo(matrix_file)
    t1 = time.time()

    A = build_symmetric_sparse(rows, cols, data, n)
    t2 = time.time()

    evals, evecs = compute_eigs_shift_invert(A, k=k, sigma=sigma, tol=tol, maxiter=maxiter, ncv=ncv)
    t3 = time.time()

    if out is None:
        out = matrix_file + ".vwmatrix"
    write_vwmatrix(out, evals, evecs)
    t4 = time.time()

    print(f"Read triplets: {len(data)} (n={n}) in {(t1-t0):.2f}s")
    print(f"Build sparse A: nnz={A.nnz} in {(t2-t1):.2f}s")
    print(f"Eigen solve: k={len(evals)}, sigma={float(sigma):.3e} in {(t3-t2):.2f}s")
    print(f"Wrote: {out} in {(t4-t3):.2f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description="Solve symmetric sparse eigenproblem and write .vwmatrix")
    ap.add_argument("matrix_file", help="Path to the 'upperhessian' file (upper-triangular COO triplets).")
    ap.add_argument("--out", default=None,
                    help="Output path. Default is '<matrix_file>.vwmatrix'.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="eigsh tolerance (default 0.0).")
    ap.add_argument("--maxiter", type=int, default=None, help="Maximum eigsh iterations (optional).")
    ap.add_argument("--ncv", type=int, default=None, help="eigsh ncv (optional).")

    # Keep overrides available for debugging, but defaults match the requested behavior.
    ap.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of eigenpairs (default {DEFAULT_K}).")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                    help=f"Constant shift sigma (default {DEFAULT_SIGMA:.9e}).")

    args = ap.parse_args()

    solve_upperhessian_to_vwmatrix(
        matrix_file=args.matrix_file,
        k=args.k,
        sigma=args.sigma,
        out=args.out,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
    )


if __name__ == "__main__":
    main()
