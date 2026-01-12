import math
import os

def _to_float(tok: str) -> float:
    # Handle Fortran D-exponent (e.g., 1.23D-04)
    return float(tok.replace("D", "E").replace("d", "E"))

def format_custom_value(val: float) -> str:
    """
    Applies the custom 9-significant figure and scientific notation logic.
    """
    if val == 0:
        return "0." + "0" * 8

    abs_val = abs(val)

    # Use scientific notation for values smaller than 10^-4
    if abs_val < 0.0001:
        # Uppercase E, no forced '+'
        return f"{val:.8E}"

    # Fixed point with ~9 significant figures
    if abs_val >= 1.0:
        digits_before = len(str(int(abs_val)))
        decimals = 9 - digits_before
        return f"{val:.{max(0, decimals)}f}"
    else:
        leading_zeros = abs(math.floor(math.log10(abs_val))) - 1
        decimals = 9 + int(leading_zeros)
        return f"{val:.{decimals}f}"

def get_row_start_spacing(is_first_row: bool, first_val_str: str) -> str:
    """
    Rule 1 & 2: Handles the specific leading spaces for rows.
    """
    if is_first_row:
        return " "  # First row always starts with one space

    # Subsequent rows:
    # If first value is negative: one space
    # If first value is positive: double space
    return " " if first_val_str.startswith("-") else "  "

def process_eigen_data(
    input_filename: str,
    output_filename: str,
    cwd: str | None = None,
    logger=None,  # pass _show_log from ui.py if you want
) -> int:
    """
    Reads eigenpairs from `input_filename` and writes a filtered + formatted vwmatrix to `output_filename`.

    - If `cwd` is provided, paths are resolved relative to cwd.
    - `logger` can be a function like _show_log(str); if None, uses print().
    - Returns new_nteig (number of kept eigenvalues).
    """
    log = logger if callable(logger) else print

    in_path  = os.path.join(cwd, input_filename) if cwd else input_filename
    out_path = os.path.join(cwd, output_filename) if cwd else output_filename

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file '{in_path}' not found.")

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        raise RuntimeError(f"Input file '{in_path}' is empty.")

    # Header (Line 1): nteig N
    header_parts = lines[0].split()
    if len(header_parts) < 2:
        raise RuntimeError(f"Bad header in '{in_path}': expected 'nteig N' on line 1.")
    nteig_orig = int(header_parts[0])
    N = int(header_parts[1])

    # Eigenvalues: next nteig_orig lines
    if len(lines) < 1 + nteig_orig:
        raise RuntimeError(
            f"File '{in_path}' too short: expected header + {nteig_orig} eigenvalue lines."
        )

    all_eigenvalues: list[tuple[float, float]] = []
    for i in range(1, nteig_orig + 1):
        parts = lines[i].split()
        if len(parts) < 2:
            raise RuntimeError(f"Bad eigenvalue line {i+1}: '{lines[i].rstrip()}'")
        all_eigenvalues.append((_to_float(parts[0]), _to_float(parts[1])))

    # Filter eigenvalues until |imag| > 1e-4
    filtered_indices: list[int] = []
    for idx, (_real, imag) in enumerate(all_eigenvalues):
        if abs(imag) > 0.0001:
            break
        filtered_indices.append(idx)

    new_nteig = len(filtered_indices)
    log(f"Filtered {nteig_orig} eigenvalues down to {new_nteig}.")

    # Eigenvectors: remaining whitespace-separated floats
    raw_vector_data = " ".join(lines[nteig_orig + 1 :]).split()
    vector_vals = [_to_float(x) for x in raw_vector_data]

    # Sanity check: do we have enough for the last filtered vector?
    if filtered_indices:
        need = (max(filtered_indices) + 1) * N
        if need > len(vector_vals):
            raise RuntimeError(
                f"Not enough eigenvector data in '{in_path}': need >= {need} floats, found {len(vector_vals)}."
            )

    # Write output
    with open(out_path, "w", encoding="utf-8") as out:
        # Row 1 (Dimensions) always starts with one space
        out.write(" " + f"{new_nteig}  {N}" + "\n")

        # Eigenvalues
        for i in range(new_nteig):
            real_part = format_custom_value(all_eigenvalues[i][0])
            imag_part = format_custom_value(all_eigenvalues[i][1])

            leading_space = get_row_start_spacing(False, real_part)
            mid_space = "  " if not imag_part.startswith("-") else " "
            out.write(f"{leading_space}{real_part}{mid_space}{imag_part}\n")

        # Eigenvectors (one vector per line)
        for j in filtered_indices:
            start_idx = j * N
            end_idx = start_idx + N
            current_vector = vector_vals[start_idx:end_idx]

            row_vals_str: list[str] = []
            for k, val in enumerate(current_vector):
                f_val = format_custom_value(val)

                if k == 0:
                    lead = get_row_start_spacing(False, f_val)
                    row_vals_str.append(lead + f_val)
                else:
                    gap = "  " if not f_val.startswith("-") else " "
                    row_vals_str.append(gap + f_val)

            out.write("".join(row_vals_str) + "\n")

    log(f"Successfully created '{out_path}'")
    return new_nteig
