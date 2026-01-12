#!/usr/bin/env python3
"""
gnm.py — Python translation of your Fortran GNM code.

Reads (in current working dir by default):
  - gnmcutoff     (single float rcut)
  - coordinates   (first line: resnum; then: <id> x y z)

Writes:
  - sortedeigen
  - sloweigenvectors
  - slowmodes
  - slow12avg
  - crosscorr
  - crosscorrslow1..crosscorrslow10
  - crosscorrslow1ext..crosscorrslow10ext

Notes:
  - Fortran uses an SVD routine (svdcmp) but your matrix is symmetric (Kirchhoff),
    so we use eigen-decomposition (numpy.linalg.eigh). Eigenvectors may differ by sign,
    but squares/correlations are equivalent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np


def read_cutoff(path: str | Path) -> float:
    txt = Path(path).read_text(encoding="utf-8").strip().split()
    if not txt:
        raise ValueError(f"Empty cutoff file: {path}")
    return float(txt[0])


def read_coordinates(path: str | Path) -> np.ndarray:
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.readline().split()
        if not head:
            raise ValueError(f"Empty coordinates file: {path}")
        n = int(head[0])
        coords = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Bad coordinates line {i+2} in {path}: {parts}")
            coords[i, 0] = float(parts[1])
            coords[i, 1] = float(parts[2])
            coords[i, 2] = float(parts[3])
    return coords


def build_kirchhoff(coords: np.ndarray, rcut: float, eps_dist: float = 1e-4) -> np.ndarray:
    """
    Fortran logic:
      cont(j,k) = -1 if r <= rcut and j!=k and r>0.0001, else 0
      cont(j,j) = -sum_k cont(j,k)
    """
    n = coords.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    rcut2 = float(rcut) * float(rcut)
    eps2 = float(eps_dist) * float(eps_dist)

    # Vectorized per-row (fast enough for typical resnum; matches Fortran O(n^2))
    for j in range(n):
        diff = coords[j] - coords  # (n,3)
        r2 = np.einsum("ij,ij->i", diff, diff)
        mask = (r2 <= rcut2) & (r2 > eps2)
        K[j, mask] = -1.0

    np.fill_diagonal(K, -K.sum(axis=1))
    return K


def write_sortedeigen(evals: np.ndarray, outpath: str | Path = "sortedeigen") -> None:
    # Fortran wrote: i, w(resnum+1-i) after descending sort -> effectively ascending list.
    with Path(outpath).open("w", encoding="utf-8") as f:
        for i, val in enumerate(evals, start=1):
            f.write(f"{i:4d}  {val:8.4f}\n")


def write_slows(evals: np.ndarray,
                evecs: np.ndarray,
                slow_idx: np.ndarray,
                nslow: int = 10) -> None:
    """
    Writes:
      sloweigenvectors: i  (v(i,slow1)..v(i,slow10))
      slowmodes:        i  (v^2 for slow1..slow10)
    """
    n = evecs.shape[0]
    use = slow_idx[:nslow]
    Vslow = evecs[:, use]  # (n, nslow)

    def fmt_row(i1: int, arr: np.ndarray) -> str:
        # Fortran: I4 then 300 f12.5
        return f"{i1:4d}" + "".join(f"{x:12.5f}" for x in arr) + "\n"

    with Path("sloweigenvectors").open("w", encoding="utf-8") as fvec, \
         Path("slowmodes").open("w", encoding="utf-8") as fmode:

        for i in range(n):
            fvec.write(fmt_row(i + 1, Vslow[i, :]))
            fmode.write(fmt_row(i + 1, Vslow[i, :] * Vslow[i, :]))


def write_slow12avg(evals: np.ndarray, evecs: np.ndarray, slow_idx: np.ndarray) -> None:
    """
    Fortran:
      s2(i) = sum_{j=slow2..slow1} v(i,j)^2 / w(j)
      top2  = sum_{j=slow2..slow1} 1 / w(j)
      slow12avg(i) = s2(i) / top2
    """
    if slow_idx.size < 2:
        raise RuntimeError("Need at least two non-zero modes for slow12avg.")

    i1 = slow_idx[0]
    i2 = slow_idx[1]
    lam1 = float(evals[i1])
    lam2 = float(evals[i2])
    v1 = evecs[:, i1]
    v2 = evecs[:, i2]

    num = (v1 * v1) / lam1 + (v2 * v2) / lam2
    den = (1.0 / lam1) + (1.0 / lam2)
    avg = num / den

    with Path("slow12avg").open("w", encoding="utf-8") as f:
        for i, val in enumerate(avg, start=1):
            # Fortran 111: I4,f12.5
            f.write(f"{i:4d}{val:12.5f}\n")


def write_crosscorr_full(evals: np.ndarray,
                         evecs: np.ndarray,
                         nz_mask: np.ndarray,
                         outpath: str | Path = "crosscorr") -> None:
    """
    Fortran computed full covariance-like matrix (pseudo-inverse) cm,
    then wrote normalized correlation: cm(i,j)/sqrt(cm(i,i)*cm(j,j))
    """
    U = evecs[:, nz_mask]  # (n, m)
    lam = evals[nz_mask]   # (m,)

    # S = U / sqrt(lam)  -> cm = S @ S.T
    S = U / np.sqrt(lam)[None, :]
    diag = np.einsum("ij,ij->i", S, S)  # cm(i,i)

    with Path(outpath).open("w", encoding="utf-8") as f:
        for i in range(S.shape[0]):
            dot = S[i, :] @ S.T  # (n,)
            denom = np.sqrt(diag[i] * diag)
            corr = np.divide(dot, denom, out=np.zeros_like(dot), where=(denom > 0.0))
            for j in range(S.shape[0]):
                # Fortran 113: i4 i4 f7.4
                f.write(f"{i+1:4d} {j+1:4d} {corr[j]:7.4f}\n")
            f.write("\n")


def write_cross_single_mode(evals: np.ndarray,
                            evecs: np.ndarray,
                            mode_vec_index: int,
                            outfile: str,
                            outfile2: str) -> None:
    """
    Matches Fortran subroutine cross:
      cm(i,j) = v(i,idx)*v(j,idx)/w(idx)  (symmetric case)
      outfile2: full normalized matrix
      outfile : only i=1 row with rl = sign(cm(1,j)) as -1/1
    """
    u = evecs[:, mode_vec_index]
    lam = float(evals[mode_vec_index])

    # normalized correlation for a single mode simplifies to sign(u_i*u_j), but we keep robust 0 handling
    ui = u[:, None]
    uj = u[None, :]
    prod = ui * uj

    denom = np.abs(ui) * np.abs(uj)
    corr = np.divide(prod, denom, out=np.zeros_like(prod), where=(denom > 0.0))

    # outfile2: full matrix
    with Path(outfile2).open("w", encoding="utf-8") as f2:
        n = u.shape[0]
        for i in range(n):
            for j in range(n):
                # Fortran 112: I4 I4 F4.1 (it’ll be -1.0, 0.0, 1.0)
                f2.write(f"{i+1:4d} {j+1:4d}   {corr[i, j]:4.1f}\n")
            f2.write("\n")

    # outfile: i=1 row with rl = sign(cm(1,j)) (same as sign(u1*u_j))
    with Path(outfile).open("w", encoding="utf-8") as f1:
        u1 = u[0]
        n = u.shape[0]
        for j in range(n):
            pj = u1 * u[j]
            rl = -1.0 if pj < 0 else (1.0 if pj > 0 else 0.0)
            f1.write(f"{1:4d} {j+1:4d}   {rl:4.1f}\n")
        f1.write("\n")


def run(coords_file: str = "coordinates",
        cutoff_file: str = "gnmcutoff",
        nslow: int = 10,
        zero_eps: float = 1e-8) -> None:

    rcut = read_cutoff(cutoff_file)
    coords = read_coordinates(coords_file)
    n = coords.shape[0]
    if n < 2:
        raise RuntimeError("resnum < 2; nothing to do.")

    K = build_kirchhoff(coords, rcut)

    # Eigen-decomposition (ascending eigenvalues)
    evals, evecs = np.linalg.eigh(K)

    write_sortedeigen(evals, "sortedeigen")

    # Non-zero modes mask (pseudo-inverse excludes ~0 eigenvalues)
    nz_mask = evals > float(zero_eps)
    nz_idx = np.where(nz_mask)[0]
    if nz_idx.size == 0:
        raise RuntimeError("All eigenvalues ~0; graph likely disconnected / cutoff too small?")

    # Slow modes = smallest non-zero eigenvalues
    slow_idx = nz_idx[:max(nslow, 2)]  # at least 2 for slow12avg
    write_slows(evals, evecs, slow_idx, nslow=nslow)
    write_slow12avg(evals, evecs, slow_idx)

    write_crosscorr_full(evals, evecs, nz_mask, outpath="crosscorr")

    # crosscorrslow{1..10} / ext
    # Fortran assumes slow1..slow10 exist; we write as many as available up to nslow.
    for m in range(1, min(nslow, slow_idx.size) + 1):
        idx = int(slow_idx[m - 1])
        write_cross_single_mode(
            evals, evecs, idx,
            outfile=f"crosscorrslow{m}",
            outfile2=f"crosscorrslow{m}ext"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="coordinates")
    ap.add_argument("--cutoff", default="gnmcutoff")
    ap.add_argument("--nslow", type=int, default=10)
    ap.add_argument("--zero-eps", type=float, default=1e-8)
    args = ap.parse_args()

    run(coords_file=args.coords, cutoff_file=args.cutoff, nslow=args.nslow, zero_eps=args.zero_eps)
