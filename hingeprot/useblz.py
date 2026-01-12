# useblz.py
from __future__ import annotations

import os
import math
import argparse
from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh


# ==========================================
# 1) FORMATTING LOGIC
# ==========================================

def format_custom_value(val: float) -> str:
    """
    Custom 9-significant figure formatting:
      - 0 -> 0.00000000
      - |val| < 1e-4 -> scientific with 8 decimals, exponent without '+'
      - otherwise -> fixed-point with ~9 significant figures
    """
    if val == 0:
        return "0." + "0" * 8

    abs_val = abs(val)

    if abs_val < 0.0001:
        # " + " removed from exponent part
        return f"{val:+.8E}".replace("+", "")

    if abs_val >= 1.0:
        digits_before = len(str(int(abs_val)))
        decimals = 9 - digits_before
        return f"{val:.{max(0, decimals)}f}"

    # abs_val in (1e-4, 1)
    leading_zeros = abs(math.floor(math.log10(abs_val))) - 1
    decimals = 9 + leading_zeros
    return f"{val:.{decimals}f}"


def get_row_start_spacing(is_first_row: bool, first_val_str: str) -> str:
    """
    Row leading spaces:
      - first row => " "
      - other rows => " " if negative else "  "
    """
    if is_first_row:
        return " "
    return " " if first_val_str.startswith("-") else "  "


# ==========================================
# 2) MATRIX READING LOGIC
# ==========================================

def read_upperhessian(filename: str) -> Tuple[csc_matrix, int]:
    """
    Reads coordinate sparse format:
      Line 1: na (number of non-zeros) [optional/ignored if not int]
      Lines 2+: i j value   (1-based indices)
    Returns:
      symmetric_mat (CSC), N
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    max_idx = -1

    with open(filename, "r") as f:
        raw_lines = f.readlines()

    # Try to interpret first line as NA; if not, treat as normal data
    start = 1
    try:
        _ = int(raw_lines[0].strip())
    except Exception:
        start = 0

    for line in raw_lines[start:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", "!", "%")):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        v = float(parts[2])

        rows.append(r)
        cols.append(c)
        data.append(v)
        if r > max_idx:
            max_idx = r
        if c > max_idx:
            max_idx = c

    if max_idx < 0:
        raise ValueError(f"No valid (i, j, val) entries found in {filename}")

    N = max_idx + 1

    # Build symmetric matrix: include (c,r) for off-diagonals
    full_rows = rows.copy()
    full_cols = cols.copy()
    full_data = data.copy()

    for r, c, v in zip(rows, cols, data):
        if r != c:
            full_rows.append(c)
            full_cols.append(r)
            full_data.append(v)

    symmetric_mat = coo_matrix((full_data, (full_rows, full_cols)), shape=(N, N)).tocsc()
    return symmetric_mat, N


# ==========================================
# 3) OUTPUT WRITER
# ==========================================

def write_vwmatrix(output_file: str, evals: np.ndarray, evecs: np.ndarray) -> None:
    """
    Writes:
      row1: " k_modes  N"
      eigenvalues: real imag(=0) each on new line
      eigenvectors: each eigenvector written sequentially, 5 numbers per line
    """
    k_modes = int(evals.shape[0])
    N = int(evecs.shape[0])

    with open(output_file, "w") as out:
        # Dimensions line
        out.write(" " + f"{k_modes}  {N}" + "\n")

        # Eigenvalues (imag part always 0.0)
        for i in range(k_modes):
            real_part = format_custom_value(float(evals[i]))
            imag_part = format_custom_value(0.0)

            leading_space = get_row_start_spacing(False, real_part)
            mid_space = "  " if not imag_part.startswith("-") else " "
            out.write(f"{leading_space}{real_part}{mid_space}{imag_part}\n")

        # Eigenvectors (columns)
        for j in range(k_modes):
            vec = evecs[:, j]
            buf: list[str] = []
            for idx, val in enumerate(vec):
                f_val = format_custom_value(float(val))

                if idx % 5 == 0:
                    if idx > 0:
                        out.write("".join(buf) + "\n")
                        buf = []
                    lead = get_row_start_spacing(False, f_val)
                    buf.append(lead + f_val)
                else:
                    gap = "  " if not f_val.startswith("-") else " "
                    buf.append(gap + f_val)

            if buf:
                out.write("".join(buf) + "\n")


# ==========================================
# 4) PUBLIC API FOR ui.py
# ==========================================

def compute_vwmatrix(
    input_file: str,
    output_file: Optional[str] = None,
    k_modes: int = 50,
    sigma: float = 1e-10,
) -> str:
    """
    Compute eigenpairs near zero using shift-invert (sigma) and write <input>.vwmatrix.

    Returns the output_file path.
    Raises exceptions instead of sys.exit() so UI can catch & display errors.
    """
    if output_file is None:
        output_file = input_file + ".vwmatrix"

    matrix, N = read_upperhessian(input_file)

    # Guard: eigsh requires 0 < k < N
    k_eff = min(int(k_modes), max(1, N - 1))

    # Solve eigenvalue problem near 0 (shift-invert)
    evals, evecs = eigsh(matrix, k=k_eff, sigma=sigma, which="LM")

    # Sort results
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    write_vwmatrix(output_file, evals, evecs)
    return output_file


# ==========================================
# 5) CLI ENTRYPOINT (kept for subprocess usage)
# ==========================================

def _main_cli() -> None:
    ap = argparse.ArgumentParser(description="Compute vwmatrix from upperhess coordinate file.")
    ap.add_argument("input_file", help="Upper Hessian coordinate file")
    ap.add_argument("--k", type=int, default=50, help="Number of modes (default: 50)")
    ap.add_argument("--sigma", type=float, default=1e-10, help="Shift-invert sigma (default: 1e-10)")
    ap.add_argument("--out", default=None, help="Output file (default: <input>.vwmatrix)")
    args = ap.parse_args()

    out = compute_vwmatrix(args.input_file, output_file=args.out, k_modes=args.k, sigma=args.sigma)
    print(out)


if __name__ == "__main__":
    _main_cli()
