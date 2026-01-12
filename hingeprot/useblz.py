# useblz.py
import argparse
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def _read_upper_tri_triples(matrix_file: str):
    """
    File format:
      - first non-comment line: na
      - next lines: i j val (1-indexed; typically upper triangle)
    We mirror off-diagonals to build a full symmetric matrix for SciPy.
    """
    rows = []
    cols = []
    data = []

    def is_comment(line: str) -> bool:
        s = line.lstrip()
        return (not s) or s.startswith(("#", "!", "c", "C"))

    with open(matrix_file, "r", encoding="utf-8", errors="ignore") as f:
        # read na
        na = None
        for line in f:
            if is_comment(line):
                continue
            parts = line.split()
            if not parts:
                continue
            na = int(parts[0])
            break

        if na is None:
            raise ValueError("Matrix file is empty or has no valid header (na).")

        read_count = 0
        nmax = -1

        for line in f:
            if read_count >= na:
                break
            if is_comment(line):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            val = float(parts[2])

            rows.append(i); cols.append(j); data.append(val)
            if i != j:
                rows.append(j); cols.append(i); data.append(val)

            nmax = max(nmax, i, j)
            read_count += 1

        if nmax < 0:
            raise ValueError("No matrix entries read (check the file content).")

        n = nmax + 1

    return (np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(data, dtype=np.float64),
            n)


def _auto_pick_k(n: int, k_user: int | None):
    """
    ANM Hessian has 6 near-zero rigid-body modes.
    We want enough eigenpairs near ~0 to cover slow modes.
    Practical default from your pipeline: 36, but NEVER >= n.
    """
    if k_user is not None and k_user > 0:
        k = int(k_user)
    else:
        k = 36  # pipeline-friendly default

    # must satisfy 1 <= k < n for eigsh
    if n <= 1:
        return 1
    k = min(k, n - 1)
    k = max(1, k)

    # if possible, try to at least include rigid-body space + 1 mode
    if n - 1 >= 7:
        k = max(k, 7)
        k = min(k, n - 1)

    return k


def solve_sparse_eigenproblem(matrix_file: str, k_user: int | None = None, sigma: float = 1e-6, output_file: str | None = None):
    print(f"Reading matrix from {matrix_file}...")
    rows, cols, data, n = _read_upper_tri_triples(matrix_file)
    print(f"Matrix Dimension N: {n}, Non-zeros (after mirroring): {len(data)}")

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    A.sum_duplicates()

    sig = float(sigma)
    # sigma=0 is risky for ANM (6 near-zero modes), keep tiny default
    if sig == 0.0:
        sig = 1e-6

    k = _auto_pick_k(n, k_user)
    print(f"Auto-selected k={k} (user k={'None' if k_user is None else k_user}), sigma={sig}")

    print(f"Solving eigenproblem (shift-invert): eigsh(k={k}, sigma={sig}) ...")
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(A, k=k, sigma=sig, which="LM")
    t1 = time.time()
    print(f"Total CPU used: {(t1 - t0)/60.0:.4f} minutes")

    # sort by eigenvalue (ascending)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if output_file is None:
        output_file = matrix_file.split(".")[0] + ".vwmatrixd"

    # Write like your Fortran-style layout:
    # first line: k n
    # then k lines: eigenvalue real, imag(0)
    # then eigenvectors in column-major (all v1, then all v2, ...)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{len(eigenvalues)} {n}\n")
        for ev in eigenvalues:
            f.write(f"{ev:20.12e} {0.0:20.12e}\n")
        for j in range(len(eigenvalues)):
            for i in range(n):
                f.write(f"{eigenvectors[i, j]:20.12e}\n")

    print(f"Results saved to {output_file}")
    return output_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_file", help="e.g. upperhessian")
    ap.add_argument("--sigma", type=float, default=1e-6, help="Shift for shift-invert (avoid 0 for ANM)")
    ap.add_argument("--k", type=int, default=None, help="Optional override. If omitted -> auto.")
    ap.add_argument("--out", default=None, help="Output .vwmatrixd path (optional)")
    args = ap.parse_args()

    solve_sparse_eigenproblem(args.matrix_file, k_user=args.k, sigma=args.sigma, output_file=args.out)


if __name__ == "__main__":
    main()
