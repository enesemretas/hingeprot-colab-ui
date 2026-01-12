# useblz.py
import argparse
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def _read_upper_tri_triples(matrix_file: str):
    """
    Reads a file with:
      first non-comment line: na  (number of triples)
      next lines: i j val   (1-indexed, upper triangle typically)
    Returns rows0, cols0, data, n (0-based, symmetric already mirrored).
    """
    rows = []
    cols = []
    data = []

    def _is_comment(line: str) -> bool:
        s = line.lstrip()
        return (not s) or s.startswith(("#", "!", "c", "C"))

    with open(matrix_file, "r", encoding="utf-8", errors="ignore") as f:
        # read na
        na = None
        for line in f:
            if _is_comment(line):
                continue
            parts = line.split()
            if not parts:
                continue
            na = int(parts[0])
            break

        if na is None:
            raise ValueError("Matrix file appears empty or has no valid header line (na).")

        # read triples
        read_count = 0
        nmax = -1
        for line in f:
            if read_count >= na:
                break
            if _is_comment(line):
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
            raise ValueError("No matrix entries read (check the file format).")

        n = nmax + 1
        return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32), np.asarray(data, dtype=np.float64), n


def solve_sparse_eigenproblem(matrix_file: str, n_eig_required: int = 36, sigma: float = 1e-6, output_file: str | None = None):
    print(f"Reading matrix from {matrix_file}...")

    rows, cols, data, n = _read_upper_tri_triples(matrix_file)
    print(f"Matrix Dimension N: {n}, Non-zeros (after mirroring): {len(data)}")

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    A.sum_duplicates()

    k = int(n_eig_required)
    if k <= 0:
        raise ValueError("k must be > 0")
    if k >= n:
        k = max(1, n - 1)

    sig = float(sigma)
    # sigma=0 is risky for ANM (6 near-zero modes) -> keep tiny default
    if sig == 0.0:
        sig = 1e-6

    print(f"Solving for {k} eigenvalues near sigma={sig} (shift-invert)...")
    start_time = time.time()

    # eigenvalues closest to sigma (original problem) using shift-invert
    eigenvalues, eigenvectors = eigsh(A, k=k, sigma=sig, which="LM")

    end_time = time.time()
    cpu_used = (end_time - start_time) / 60.0
    print(f"Total CPU used: {cpu_used:.4f} minutes")

    # sort ascending like typical “slowest first” view for ANM around 0
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Save results to .vwmatrixd (your layout)
    if output_file is None:
        output_file = matrix_file.split(".")[0] + ".vwmatrixd"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{len(eigenvalues)} {n}\n")

        # Write eigenvalues (imag part 0)
        for ev in eigenvalues:
            f.write(f"{ev:20.12e} {0.0:20.12e}\n")

        # Write eigenvectors column-major (all of v1 then all of v2...)
        for j in range(len(eigenvalues)):
            for i in range(n):
                f.write(f"{eigenvectors[i, j]:20.12e}\n")

    print(f"Results saved to {output_file}")
    return output_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_file", help="Path to upper-triangular triple file, e.g. upperhessian")
    ap.add_argument("--k", type=int, default=36, help="Number of eigenpairs to compute")
    ap.add_argument("--sigma", type=float, default=1e-6, help="Shift for shift-invert (avoid 0 for ANM)")
    ap.add_argument("--out", default=None, help="Output .vwmatrixd path (optional)")
    args = ap.parse_args()

    solve_sparse_eigenproblem(args.matrix_file, n_eig_required=args.k, sigma=args.sigma, output_file=args.out)


if __name__ == "__main__":
    main()
