#!/usr/bin/env python3
"""
useblz_py.py — Python replacement for the Fortran "useblz" driver.

Input matrix format (same as Fortran code):
  first line: na  (declared number of nonzeros; may be approximate)
  then lines: irn jcn a  (1-based indices), storing ONLY upper triangle (i<=j)

Output format (matches the Fortran write pattern):
  line 1: NTEIG N
  next NTEIG lines: eig(i)  err(i)
  then all eigenvector entries x(i,j) in Fortran column-major order (i fastest)
  written with scientific format and wrapped (5 numbers per line).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, eigsh, splu


def _read_upper_tri_coo(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Stream-read the file efficiently.
    Returns (rows0, cols0, data0, n) with 0-based indices and n = max index + 1.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Read first non-empty, non-comment line as "na"
        first = ""
        while True:
            first = f.readline()
            if first == "":
                raise ValueError("Empty file.")
            s = first.strip()
            if s and not s.startswith(("#", "!", "c", "C")):
                break

        # declared nnz (Fortran ignores it, but we use it as initial capacity)
        try:
            declared_na = int(s.split()[0])
        except Exception as e:
            raise ValueError(f"Failed to parse first line as integer na: {s!r}") from e

        cap = max(declared_na, 1024)
        rows = np.empty(cap, dtype=np.int32)
        cols = np.empty(cap, dtype=np.int32)
        data = np.empty(cap, dtype=np.float64)

        k = 0
        nmax = 0

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

            # Grow if needed
            if k >= cap:
                new_cap = int(cap * 1.5) + 1
                rows2 = np.empty(new_cap, dtype=rows.dtype)
                cols2 = np.empty(new_cap, dtype=cols.dtype)
                data2 = np.empty(new_cap, dtype=data.dtype)
                rows2[:k] = rows[:k]
                cols2[:k] = cols[:k]
                data2[:k] = data[:k]
                rows, cols, data = rows2, cols2, data2
                cap = new_cap

            rows[k] = i - 1  # to 0-based
            cols[k] = j - 1
            data[k] = a
            k += 1

            if i > nmax:
                nmax = i
            if j > nmax:
                nmax = j

        rows = rows[:k]
        cols = cols[:k]
        data = data[:k]
        n = nmax
        if n <= 0:
            raise ValueError("Matrix size detected as <= 0 (check input indices).")
        return rows, cols, data, n


def _build_symmetric_sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int) -> csr_matrix:
    """
    Input contains only upper triangle. Build A = A_upper + A_upper^T - diag(A_upper).
    """
    off = rows != cols
    # Duplicate off-diagonal entries symmetrically
    rr = np.concatenate([rows, cols[off]])
    cc = np.concatenate([cols, rows[off]])
    dd = np.concatenate([data, data[off]])

    A = coo_matrix((dd, (rr, cc)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def _write_vwmatrixd(out_path: str, evals: np.ndarray, errs: np.ndarray, evecs: np.ndarray) -> None:
    """
    Matches the Fortran structure:
      NTEIG N
      eig err (NTEIG lines)
      then x(i,j) in Fortran column-major order
    """
    n, k = evecs.shape
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{k} {n}\n")
        for lam, err in zip(evals, errs):
            f.write(f"{lam:.16e} {err:.16e}\n")

        flat = evecs.reshape(n * k, order="F")  # column-major like Fortran ((x(i,j),i=1,N),j=1,k)

        per_line = 5
        for idx, val in enumerate(flat, start=1):
            f.write(f"{val:.16e}")
            if idx % per_line == 0:
                f.write("\n")
            else:
                f.write(" ")
        if (n * k) % per_line != 0:
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrixfile", help="Input sparse matrix file (upper-triangular coordinate format).")
    ap.add_argument("-k", "--nreig", type=int, default=41, help="Number of eigenpairs to compute (default: 41).")
    ap.add_argument("--sigma", type=float, default=None, help="Shift σ for shift-invert (eigenvalues near σ).")
    ap.add_argument("--eigl", type=float, default=None, help="Optional interval lower bound; used only to pick σ.")
    ap.add_argument("--eigr", type=float, default=None, help="Optional interval upper bound; used only to pick σ.")
    ap.add_argument("--tol", type=float, default=0.0, help="ARPACK tolerance (0.0 lets ARPACK choose).")
    ap.add_argument("--maxiter", type=int, default=None, help="Max ARPACK iterations (default: SciPy/ARPACK default).")
    ap.add_argument("--ncv", type=int, default=None, help="ARPACK subspace size (optional).")
    args = ap.parse_args()

    t0 = time.time()

    rows, cols, data, n = _read_upper_tri_coo(args.matrixfile)
    A = _build_symmetric_sparse(rows, cols, data, n)  # CSR for fast matvec
    A_csc = A.tocsc()  # CSC for factorization

    # Choose sigma (mimics “use EIGL/EIGR to pick a shift” idea)
    sigma = args.sigma
    if sigma is None and args.eigl is not None and args.eigr is not None and args.eigr != args.eigl:
        sigma = 0.5 * (args.eigl + args.eigr)
    if sigma is None:
        sigma = 0.0  # sensible default if you want eigenvalues near 0

    k = int(args.nreig)
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < n. Got k={k}, n={n}.")

    # Build (A - sigma I) factorization for fast repeated solves (shift-invert)
    Ashift = (A_csc - sigma * eye(n, format="csc", dtype=np.float64))

    try:
        lu = splu(Ashift)  # SuperLU (general sparse LU); good practical replacement for MA47 here
    except RuntimeError as e:
        raise RuntimeError(
            "Factorization of (A - sigma*I) failed. "
            "Try a different --sigma (avoid being exactly on an eigenvalue), "
            "or add a tiny perturbation (e.g., sigma += 1e-8)."
        ) from e

    def op_solve(x: np.ndarray) -> np.ndarray:
        return lu.solve(x)

    OPinv = LinearOperator((n, n), matvec=op_solve, dtype=np.float64)

    # Compute eigenpairs near sigma
    evals, evecs = eigsh(
        A,
        k=k,
        sigma=sigma,
        which="LM",
        OPinv=OPinv,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
        return_eigenvectors=True,
    )

    # Sort ascending (BLZPACK often returns in an order depending on shift)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # Residual-based "err" analogous to EIG(i,2)
    # err_i = ||A v - λ v|| / (|λ| ||v|| + eps)
    Av = A.dot(evecs)
    eps = 1e-30
    errs = np.empty(k, dtype=np.float64)
    for i in range(k):
        r = Av[:, i] - evals[i] * evecs[:, i]
        num = np.linalg.norm(r)
        den = abs(evals[i]) * np.linalg.norm(evecs[:, i]) + eps
        errs[i] = num / den

    base, ext = os.path.splitext(args.matrixfile)
    out_path = base + ".vwmatrixd" if base else (args.matrixfile + ".vwmatrixd")
    _write_vwmatrixd(out_path, evals, errs, evecs)

    t1 = time.time()
    print(f"Read nnz={len(data):,}, n={n:,}")
    print(f"Computed k={k} eigenpairs near sigma={sigma}")
    print(f"Wrote: {out_path}")
    print(f"Total wall time: {(t1 - t0)/60.0:.2f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
