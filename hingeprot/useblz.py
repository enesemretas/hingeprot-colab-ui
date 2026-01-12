#!/usr/bin/env python3
"""useblz.py

SciPy-based replacement for BLZPACK/MA47 eigen solve for ANM.

Input: upper-triangular coordinate (triplet) file (Fortran-style 1-indexed):
  line 1:  NA            (declared number of stored triplets)
  next lines:  i  j  a_ij (1-indexed indices; usually only upper triangle)

Output (default): <input>.vwmatrix
  line 1:  (one leading space) "k n"
  next k lines:  eigenvalue  imag_part   (imag_part is written as 0.0)
  then:  eigenvectors, written in Fortran column-major order (all of v1, then v2, ...),
         5 numbers per line.

Formatting rules (per user request):
  1) Header line starts with exactly one space.
  2) Other lines start with one leading space if first number is negative,
     otherwise two leading spaces.
  3) Between numbers on the same line:
       - if the next number is negative -> 1 space before it
       - else -> 2 spaces before it
  4) Numbers use 9 significant figures in fixed format until 0.0001; below that -> scientific,
     with mantissa having 8 decimals and an 'E' exponent (e.g., -3.30213541E-07).
  5) Eigenvectors are globally sign-flipped: eigenvectors = -eigenvectors.

Sigma rule (per user request):
  - sigma is machine epsilon by default (np.finfo(float).eps), unless overridden via --sigma.

Usage:
  python3 useblz.py upperhessian
  python3 useblz.py upperhessian --out upperhessian.vwmatrix
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Tuple

import numpy as np


def read_upper_tri_coo(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read upper-triangle triplets (i, j, a_ij) and return 0-based COO arrays + n."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # first non-empty, non-comment line should be NA
        first = ""
        while True:
            first = f.readline()
            if first == "":
                raise ValueError("Empty matrix file.")
            s = first.strip()
            if s and not s.startswith(("#", "!", "c", "C")):
                break

        try:
            _na_declared = int(s.split()[0])
        except Exception as e:
            raise ValueError(f"Failed to parse first line as integer NA: {s!r}") from e

        nmax = 0
        for line in f:
            line = line.strip()
            if not line or line[0] in "#!":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            a = float(parts[2])
            rows.append(i)
            cols.append(j)
            data.append(a)
            nmax = max(nmax, i + 1, j + 1)

    if nmax <= 0 or len(rows) == 0:
        raise ValueError("No valid triplets were read (check file format).")

    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
        int(nmax),
    )


def build_symmetric_sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int):
    """Build symmetric A from upper-tri COO."""
    from scipy.sparse import coo_matrix

    off = rows != cols
    rr = np.concatenate([rows, cols[off]])
    cc = np.concatenate([cols, rows[off]])
    dd = np.concatenate([data, data[off]])
    A = coo_matrix((dd, (rr, cc)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def auto_k(n: int, desired_nonzero_modes: int = 35) -> int:
    """Auto-select eigenpair count: k = desired_nonzero_modes + 6 (clamped to [1, n-1])."""
    k = int(desired_nonzero_modes) + 6
    if n <= 1:
        return 1
    if k >= n:
        k = n - 1
    return max(1, k)


def auto_sigma_machine_epsilon() -> float:
    """Sigma is machine epsilon by default."""
    return float(np.finfo(np.float64).eps)


def format_9sig(x: float) -> str:
    """Format number using the requested 9-significant-figure rules."""
    if x == 0.0:
        return "0.000000000"

    ax = abs(x)

    # switch to scientific at 1e-4 and below (E-05 or smaller)
    if ax < 1e-4:
        return f"{x:.8E}"

    # fixed-point
    if ax >= 1.0:
        int_digits = int(math.floor(math.log10(ax))) + 1
        dec = max(0, 9 - int_digits)
        return f"{x:.{dec}f}"

    # 0 < ax < 1: count leading zeros after decimal (0 for [0.1,1), 1 for [0.01,0.1), ...)
    z = int(math.floor(-math.log10(ax) - 1e-12))
    if z > 3:
        return f"{x:.8E}"
    dec = 9 + z
    return f"{x:.{dec}f}"


def format_line(nums: list[float], header: bool = False) -> str:
    """Apply spacing rules to a list of numbers."""
    if header:
        # exactly one leading space
        return " " + " ".join(str(int(v)) for v in nums)

    # For every value in the row:
    # - if negative: 1 leading space before the number
    # - if positive: 2 leading spaces before the number
    out = []
    for v in nums:
        s = format_9sig(float(v))
        out.append((" " if float(v) < 0.0 else "  ") + s)
    return "".join(out)


def solve_upperhessian_to_vwmatrix(
    upper_path: str,
    out_path: Optional[str] = None,
    k: Optional[int] = None,
    sigma: Optional[float] = None,
    desired_nonzero_modes: int = 35,
) -> str:
    """Compute eigenpairs near sigma and write .vwmatrix."""
    from scipy.sparse import eye
    from scipy.sparse.linalg import eigsh, splu, LinearOperator

    rows, cols, data, n = read_upper_tri_coo(upper_path)
    A = build_symmetric_sparse(rows, cols, data, n)
    A_csc = A.tocsc()

    k_eff = auto_k(n, desired_nonzero_modes) if (k is None or k <= 0) else int(k)

    # sigma: machine epsilon by default (per request)
    sig = auto_sigma_machine_epsilon() if (sigma is None) else float(sigma)

    # Robust factorization: if sigma too close to an eigenvalue, nudge it
    lu = None
    last_err = None
    for _ in range(8):
        try:
            Ashift = (A_csc - sig * eye(n, format="csc", dtype=np.float64))
            lu = splu(Ashift)
            break
        except Exception as e:
            last_err = e
            sig *= 10.0
    if lu is None:
        raise RuntimeError(f"Factorization failed; last error: {last_err}")

    OPinv = LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)

    evals, evecs = eigsh(
        A,
        k=k_eff,
        sigma=sig,
        which="LM",
        OPinv=OPinv,
        tol=0.0,
        maxiter=None,
        ncv=None,
        return_eigenvectors=True,
    )

    # Sort ascending eigenvalues
    order = np.argsort(evals)
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)

    # ---- requested change: flip ALL eigenvectors globally ----
    evecs = -evecs

    if out_path is None:
        out_path = upper_path + ".vwmatrix"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(format_line([k_eff, n], header=True) + "\n")

        # eigenvalues: (real, imag) ; imag is 0.0
        for lam in evals:
            f.write(format_line([float(lam), 0.0]) + "\n")

        # eigenvectors in Fortran order, 5 values/line
        flat = evecs.reshape(n * k_eff, order="F")
        per_line = 5
        for i in range(0, flat.size, per_line):
            f.write(format_line(flat[i : i + per_line].tolist()) + "\n")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Compute ANM eigenpairs from upperhessian and write .vwmatrix.")
    ap.add_argument("upperhessian", help="Path to upperhessian triplet file")
    ap.add_argument("--out", default=None, help="Output .vwmatrix path (default: <input>.vwmatrix)")
    ap.add_argument("--k", type=int, default=0, help="Eigenpair count (0 => auto)")
    ap.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Shift sigma (default: machine epsilon). If factorization fails, it will be nudged upward.",
    )
    ap.add_argument("--nonzero-modes", type=int, default=35, help="Desired nonzero modes (auto k = this + 6)")
    args = ap.parse_args()

    outp = solve_upperhessian_to_vwmatrix(
        upper_path=args.upperhessian,
        out_path=args.out,
        k=args.k,
        sigma=args.sigma,
        desired_nonzero_modes=args.nonzero_modes,
    )
    print(f"Wrote: {outp}")


if __name__ == "__main__":
    main()
