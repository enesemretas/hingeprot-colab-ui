#!/usr/bin/env python3
"""
useblz.py
---------

SciPy-based replacement for BLZPACK/MA47 eigen solve.

Input: upper-triangular coordinate (triplet) file (Fortran-style 1-indexed):
  line 1:  NA            (declared number of stored triplets)
  next lines:  i  j  a_ij (1-indexed indices; usually only upper triangle)

Output (default): .vwmatrix (in the same directory as the input matrix)
  line 1:  (ONE leading space) "k n"
  next k lines:  eigenvalue  imag_part   (imag_part is written as 0.0)
  then:  eigenvectors, written in Fortran column-major order (all of v1, then v2, ...),
         5 numbers per line.

Automatic settings (per request):
  - k = 38
  - sigma = 2.22044605E-16 (machine epsilon), CONSTANT (no auto-adjust)

Formatting rules (per request):
  1) Header line starts with exactly one space.
  2) Every other line:
       - starts with one leading space if the first number is negative
       - starts with two leading spaces otherwise
  3) Each value contains 9 significant figures in fixed format until |x| < 1e-4,
     then scientific with 8 decimals and 'E' exponent (e.g., -3.30213541E-07).
     For 0 < |x| < 1, decimals increase with leading zeros after decimal:
       0.711324482  -> 9 decimals
       0.0308059661 -> 10 decimals
       0.0030...    -> 11 decimals
       0.0003...    -> 12 decimals
     After that (or below 1e-4) -> scientific.
  4) Between numbers on the same line:
       - if the next number is negative -> 1 space before it
       - else -> 2 spaces before it

NOTE: Eigenvector signs are NOT modified.

Usage:
  python3 useblz.py upperhessian
  python3 useblz.py upperhessian --out upperhessian.vwmatrix
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Tuple

import numpy as np

DEFAULT_K = 38
DEFAULT_SIGMA_STR = "2.22044605E-16"
DEFAULT_SIGMA = float(DEFAULT_SIGMA_STR)
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
    rows = []
    cols = []
    data = []

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


def compute_eigs_shift_invert(
    A,
    k: int,
    sigma: float,
    tol: float = DEFAULT_TOL,
    maxiter: int | None = None,
    ncv: int | None = None,
):
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
        raise ValueError("sigma must be non-zero. Requested sigma was 0.0.")

    # Factorize (A - sigma I) ONCE (constant sigma: no automatic adjustment)
    Ashift = (A - sig * eye(n, format="csc", dtype=np.float64))
    try:
        lu = splu(Ashift)
    except Exception as e:
        raise RuntimeError(
            f"Shifted factorization failed for sigma={sig}. "
            f"(You requested constant sigma={DEFAULT_SIGMA_STR}). Original error: {e}"
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
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs


# ---------- formatting helpers (your .vwmatrix rules) ----------
def format_9sig(x: float) -> str:
    """9 significant figures in fixed, switch to scientific when |x| < 1e-4."""
    if x == 0.0:
        return "0.000000000"

    ax = abs(x)

    # scientific at 1e-4 and below
    if ax < 1e-4:
        return f"{x:.8E}"

    # fixed-point for |x| >= 1.0
    if ax >= 1.0:
        int_digits = int(math.floor(math.log10(ax))) + 1
        dec = max(0, 9 - int_digits)
        return f"{x:.{dec}f}"

    # 0 < |x| < 1: add decimals based on leading zeros after decimal, up to 3
    z = int(math.floor(-math.log10(ax) - 1e-12))  # 0 for [0.1,1), 1 for [0.01,0.1), ...
    if z > 3:
        return f"{x:.8E}"
    dec = 9 + z
    return f"{x:.{dec}f}"


def format_line(nums: list[float], header: bool = False) -> str:
    """
    Header:
      - exactly one leading space, then "k n"
    Other:
      - leading: 1 space if first value negative else 2
      - separators: 1 space before negative values, else 2
    """
    if header:
        # ensure ints for k and n
        return " " + f"{int(nums[0])} {int(nums[1])}"

    if not nums:
        return ""

    s0 = format_9sig(float(nums[0]))
    line = (" " if nums[0] < 0 else "  ") + s0

    for v in nums[1:]:
        sv = format_9sig(float(v))
        line += (" " if v < 0 else "  ") + sv

    return line


def write_vwmatrix(out_path: str, evals: np.ndarray, evecs: np.ndarray):
    """
    Write BLZPACK-like .vwmatrix file with your spacing/precision rules.
      (one leading space) k n
      k lines: eigenvalue imag_part (0.0)
      eigenvectors in Fortran column-major order, 5 values per line
    """
    n, k = evecs.shape
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(format_line([k, n], header=True) + "\n")

        for lam in evals:
            f.write(format_line([float(lam), 0.0], header=False) + "\n")

        flat = np.asarray(evecs, dtype=np.float64).reshape(n * k, order="F")
        per_line = 5
        for i in range(0, flat.size, per_line):
            chunk = flat[i : i + per_line].tolist()
            f.write(format_line(chunk, header=False) + "\n")


def solve_upperhessian_to_vwmatrix(
    matrix_file: str,
    out: str | None = None,
    tol: float = DEFAULT_TOL,
    maxiter: int | None = None,
    ncv: int | None = None,
) -> str:
    t0 = time.time()
    rows, cols, data, n = read_upper_tri_coo(matrix_file)
    t1 = time.time()

    A = build_symmetric_sparse(rows, cols, data, n)
    t2 = time.time()

    # Automatic settings (per request)
    k = DEFAULT_K
    sigma = DEFAULT_SIGMA

    evals, evecs = compute_eigs_shift_invert(A, k=k, sigma=sigma, tol=tol, maxiter=maxiter, ncv=ncv)
    t3 = time.time()

    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(matrix_file)), ".vwmatrix")

    write_vwmatrix(out, evals, evecs)
    t4 = time.time()

    print(f"Read triplets: {len(data)} (n={n}) in {(t1 - t0):.2f}s")
    print(f"Build sparse A: nnz={A.nnz} in {(t2 - t1):.2f}s")
    print(f"Eigen solve: k={len(evals)}, sigma={DEFAULT_SIGMA_STR} in {(t3 - t2):.2f}s")
    print(f"Wrote: {out} in {(t4 - t3):.2f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description="Solve symmetric sparse eigenproblem and write .vwmatrix")
    ap.add_argument("matrix_file", help="Path to the 'upperhessian' file (upper-triangular COO triplets).")
    ap.add_argument("--out", default=None, help="Output path. Default is '.vwmatrix' in the same folder as the input.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="eigsh tolerance (default 0.0).")
    ap.add_argument("--maxiter", type=int, default=None, help="Maximum eigsh iterations (optional).")
    ap.add_argument("--ncv", type=int, default=None, help="eigsh ncv (optional).")
    args = ap.parse_args()

    solve_upperhessian_to_vwmatrix(
        matrix_file=args.matrix_file,
        out=args.out,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
    )


if __name__ == "__main__":
    main()
