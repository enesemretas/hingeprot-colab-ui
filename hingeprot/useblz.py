# useblz.py
from __future__ import annotations

import os
import math
import argparse
from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import eigsh  # keeps your current solver


# ============================================================
# 1) FORMAT RULES YOU SPECIFIED
# ============================================================

def format_custom_value(val: float) -> str:
    """
    9 significant-figure style formatting with your rules:

    - If val == 0 -> 0.00000000 (fixed)
    - If |val| < 1e-4 -> scientific with 8 decimals, E±xx, no '+' sign
      examples: -3.30213541E-07, 2.22044605E-16
    - Else fixed-point with variable decimals to keep ~9 significant digits:
        * abs>=1: total sig digits 9 => decimals = max(0, 9 - digits_before_decimal)
        * abs<1: decimals = 9 + (#leading_zeros_after_decimal)
          e.g. 0.711... -> 9 decimals
               0.0308.. -> 10 decimals
               0.00308. -> 11 decimals
               0.000308 -> (would be scientific because <1e-4)
    """
    if val == 0 or val == 0.0:
        return "0.00000000"

    abs_val = abs(val)

    # Switch to scientific after 0.0001 threshold (>= 4 zeros after decimal)
    if abs_val < 1e-4:
        s = f"{val:.8E}"  # 8 decimals
        # remove "+" in exponent if present
        s = s.replace("E+","E").replace("e+","E").replace("e","E")
        return s

    # Fixed point with ~9 significant digits
    if abs_val >= 1.0:
        digits_before = len(str(int(abs_val)))
        decimals = 9 - digits_before
        if decimals < 0:
            decimals = 0
        return f"{val:.{decimals}f}"

    # 0 < abs_val < 1: count leading zeros after decimal to decide decimals
    # Example:
    # 0.711... => leading_zeros=0 => decimals=9
    # 0.03.... => leading_zeros=1 => decimals=10
    # 0.003... => leading_zeros=2 => decimals=11
    leading_zeros = int(abs(math.floor(math.log10(abs_val))) - 1)
    decimals = 9 + leading_zeros
    return f"{val:.{decimals}f}"


def row_prefix_for_first_value(first_val_str: str, is_header: bool = False) -> str:
    """
    Rule:
      - Header row starts with ONE space.
      - Rows 2+:
          if first value is negative -> ONE space
          else -> TWO spaces
    """
    if is_header:
        return " "
    return " " if first_val_str.startswith("-") else "  "


def gap_before_value(val_str: str) -> str:
    """
    Rule 4:
      - if value is negative -> one space before it
      - else -> two spaces before it
    (This is the spacing *between* values within the same row.)
    """
    return " " if val_str.startswith("-") else "  "


# ============================================================
# 2) READ UPPERHESSIAN (COO upper-triangle) -> symmetric CSC
# ============================================================

def read_upperhessian(filename: str) -> Tuple[csc_matrix, int]:
    """
    Reads coordinate sparse format:
      Line 1: na (optional integer count)
      Next lines: i j value   (1-based indices)

    Builds a symmetric matrix by mirroring off-diagonal entries.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r") as f:
        raw_lines = f.readlines()

    start = 1
    try:
        _ = int(raw_lines[0].strip())
    except Exception:
        start = 0

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    max_idx = -1

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
        max_idx = max(max_idx, r, c)

    if max_idx < 0:
        raise ValueError(f"No valid (i, j, val) entries found in {filename}")

    N = max_idx + 1

    full_rows = rows.copy()
    full_cols = cols.copy()
    full_data = data.copy()

    for r, c, v in zip(rows, cols, data):
        if r != c:
            full_rows.append(c)
            full_cols.append(r)
            full_data.append(v)

    mat = coo_matrix((full_data, (full_rows, full_cols)), shape=(N, N)).tocsc()
    return mat, N


# ============================================================
# 3) WRITE upperhessian.vwmatrixd EXACTLY AS YOU DESCRIBED
# ============================================================

def write_vwmatrixd(
    output_file: str,
    evals: np.ndarray,
    evecs: np.ndarray,
    imag_tol: float = 1e-4,
    per_line: int = 5,
) -> int:
    """
    Writes:
      Row 1: " [nteig]  [N]"  (header starts with one space)
      Next nteig rows: eigenvalues real imag
      Remaining: eigenvectors column-major (vec1 all N, then vec2 all N, ...),
                 wrapped to 'per_line' values per row (line-wrap does not change parsing).

    Selection rule:
      nteig is chosen by scanning evals in order and STOPPING at the first eigenvalue
      with |Imag| > imag_tol.
    """
    # Ensure complex-safe arrays
    evals_c = np.asarray(evals, dtype=np.complex128)
    evecs_c = np.asarray(evecs, dtype=np.complex128)

    N = int(evecs_c.shape[0])
    k_avail = int(evecs_c.shape[1])

    # Determine nteig by "until imag part becomes higher than 1e-4"
    nteig = 0
    for i in range(min(len(evals_c), k_avail)):
        if abs(evals_c[i].imag) > imag_tol:
            break
        nteig += 1

    if nteig <= 0:
        raise RuntimeError("No eigenvalues passed the imaginary-part threshold (|Imag| <= 1e-4).")

    # Slice to selected
    evals_sel = evals_c[:nteig]
    evecs_sel = evecs_c[:, :nteig]

    with open(output_file, "w") as out:
        # ---------- HEADER ----------
        header = f"{nteig}  {N}"
        out.write(row_prefix_for_first_value(header, is_header=True) + header + "\n")

        # ---------- EIGENVALUES ----------
        for i in range(nteig):
            real_str = format_custom_value(float(evals_sel[i].real))
            imag_str = format_custom_value(float(evals_sel[i].imag))

            # Row start depends on sign of FIRST column (real)
            line = row_prefix_for_first_value(real_str, is_header=False) + real_str
            # spacing before imag depends on imag sign
            line += gap_before_value(imag_str) + imag_str
            out.write(line + "\n")

        # ---------- EIGENVECTORS (column-major) ----------
        # Fortran: ((x(i,j),i=1,N),j=1,nteig) => column-major flatten
        # Use REAL part (ANM symmetric should be real; if complex, this matches "x" being real in Fortran)
        flat = np.asarray(evecs_sel.real, dtype=np.float64).ravel(order="F")

        # Write with your spacing rules and wrapping
        idx = 0
        total = flat.size
        while idx < total:
            chunk = flat[idx : min(idx + per_line, total)]
            # first value decides row prefix
            first_str = format_custom_value(float(chunk[0]))
            line = row_prefix_for_first_value(first_str, is_header=False) + first_str
            for v in chunk[1:]:
                s = format_custom_value(float(v))
                line += gap_before_value(s) + s
            out.write(line + "\n")
            idx += per_line

    return nteig


# ============================================================
# 4) MAIN: solve + sort + write
# ============================================================

def compute_and_write(
    input_file: str,
    output_file: Optional[str] = None,
    imag_tol: float = 1e-4,
    per_line: int = 5,
    also_write_vwmatrix: bool = True,
) -> str:
    """
    Solves eigenproblem (kept as eigsh like your current pipeline),
    sorts by eigenvalue (ascending), then writes upperhessian.vwmatrixd.

    If also_write_vwmatrix=True, also writes a copy <input>.vwmatrix
    so your existing ui.py doesn't break.
    """
    if output_file is None:
        output_file = input_file + ".vwmatrixd"

    mat, N = read_upperhessian(input_file)

    # Keep your existing behavior: eigsh (symmetric).
    # (Imag parts will be 0, but we still implement the imag_tol selection exactly.)
    k = min(60, max(2, N - 1))  # a safe default; selection is handled after
    evals, evecs = eigsh(mat, k=k, sigma=0.0, which="LM")

    # Sort by eigenvalue (ascending real part)
    order = np.argsort(np.real(evals))
    evals = evals[order]
    evecs = evecs[:, order]

    write_vwmatrixd(output_file, evals, evecs, imag_tol=imag_tol, per_line=per_line)

    if also_write_vwmatrix:
        compat = input_file + ".vwmatrix"
        if compat != output_file:
            # write the same content to .vwmatrix too
            # (ui.py currently looks for upperhessian.vwmatrix)
            # simplest: copy by re-writing using same already-solved eval/evec
            write_vwmatrixd(compat, evals, evecs, imag_tol=imag_tol, per_line=per_line)

    return output_file


def _main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="e.g. upperhessian")
    ap.add_argument("--out", default=None, help="Default: <input>.vwmatrixd")
    ap.add_argument("--imag_tol", type=float, default=1e-4)
    ap.add_argument("--per_line", type=int, default=5)
    ap.add_argument("--no_compat", action="store_true", help="Do NOT also write <input>.vwmatrix")
    args = ap.parse_args()

    out = compute_and_write(
        args.input_file,
        output_file=args.out,
        imag_tol=args.imag_tol,
        per_line=args.per_line,
        also_write_vwmatrix=(not args.no_compat),
    )
    print(out)


if __name__ == "__main__":
    _main_cli()
