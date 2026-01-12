import math
import os

def format_custom_value(val):
    """
    Applies the custom 9-significant figure and scientific notation logic.
    """
    if val == 0:
        return "0." + "0" * 8
    
    abs_val = abs(val)
    
    # Rule 3: Use scientific notation for values smaller than 10^-4 (E-05 and smaller)
    if abs_val < 0.0001:
        # Format as: [-]X.XXXXXXXXE-XX (1 digit before dot, 8 after = 9 sig figs)
        # We manually construct this to ensure uppercase 'E' and correct spacing
        formatted = f"{val:+.8E}".replace('+', '')
        return formatted
    
    # Rule 3: Fixed point with 9 significant figures
    if abs_val >= 1.0:
        # digits_before_dot is the number of digits left of the decimal
        digits_before = len(str(int(abs_val)))
        decimals = 9 - digits_before
        return f"{val:.{max(0, decimals)}f}"
    else:
        # For values < 1 (e.g., 0.711..., 0.0308...)
        # Calculate leading zeros after the decimal point
        leading_zeros = abs(math.floor(math.log10(abs_val))) - 1
        decimals = 9 + int(leading_zeros)
        return f"{val:.{decimals}f}"

def get_row_start_spacing(is_first_row, first_val_str):
    """
    Rule 1 & 2: Handles the specific leading spaces for rows.
    """
    if is_first_row:
        return " " # First row always starts with one space
    
    # Subsequent rows: 
    # If first value is negative: one space
    # If first value is positive: double space
    if first_val_str.startswith('-'):
        return " "
    else:
        return "  "

def process_eigen_data(input_filename, output_filename):
    # Check if input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    # 1. Read the input file
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("Error: Input file is empty.")
        return

    # 2. Parse the Header (Line 1)
    header_parts = lines[0].split()
    nteig_orig = int(header_parts[0])
    N = int(header_parts[1])

    # 3. Parse Eigenvalues (Lines 2 to nteig+1)
    # Stored as tuples: (Real, Imag)
    all_eigenvalues = []
    # Note: Fortran output usually has fixed width, but split() handles variable whitespace well
    for i in range(1, nteig_orig + 1):
        parts = lines[i].split()
        if len(parts) >= 2:
            all_eigenvalues.append((float(parts[0]), float(parts[1])))

    # 4. Filter Logic: 
    # Select eigenvalues until the Imaginary part > 10^-4
    filtered_indices = []
    for idx, (real, imag) in enumerate(all_eigenvalues):
        if abs(imag) > 0.0001:
            break
        filtered_indices.append(idx)
    
    new_nteig = len(filtered_indices)
    print(f"Filtered {nteig_orig} eigenvalues down to {new_nteig}.")

    # 5. Parse Eigenvectors (Remaining lines)
    # The Fortran code writes ((x(i,j), i=1,N), j=1,nteig)
    # This means all N elements of vector 1, then all N of vector 2, etc.
    # We join all remaining lines into one long string and split by whitespace
    raw_vector_data = " ".join(lines[nteig_orig + 1:]).split()
    vector_vals = [float(x) for x in raw_vector_data]

    # 6. Write the Output File
    with open(output_filename, 'w') as out:
        
        # --- WRITE ROW 1 (Dimensions) ---
        # Rule 1: First row starts with ONE space
        dim_str = f"{new_nteig}  {N}"
        out.write(" " + dim_str + "\n") 

        # --- WRITE EIGENVALUES ---
        for i in range(new_nteig):
            real_part = format_custom_value(all_eigenvalues[i][0])
            imag_part = format_custom_value(all_eigenvalues[i][1])
            
            # Rule 2: Row start spacing
            leading_space = get_row_start_spacing(False, real_part)
            
            # Rule 4: Internal spacing
            # If next number is negative, 1 space (gap + minus = 2 chars visual)
            # If next number is positive, 2 spaces
            mid_space = "  " if not imag_part.startswith('-') else " "
            
            out.write(f"{leading_space}{real_part}{mid_space}{imag_part}\n")

        # --- WRITE EIGENVECTORS ---
        # We output ONLY the eigenvectors corresponding to the filtered eigenvalues
        for j in filtered_indices:
            # Calculate the slice for this vector
            start_idx = j * N
            end_idx = start_idx + N
            
            if end_idx > len(vector_vals):
                print(f"Warning: Not enough data for eigenvector {j+1}")
                break
                
            current_vector = vector_vals[start_idx : end_idx]
            
            row_vals_str = []
            for k, val in enumerate(current_vector):
                f_val = format_custom_value(val)
                
                if k == 0:
                    # Starting the row (first column)
                    lead = get_row_start_spacing(False, f_val)
                    row_vals_str.append(lead + f_val)
                else:
                    # Spacing between values in the same row (Rule 4)
                    gap = "  " if not f_val.startswith('-') else " "
                    row_vals_str.append(gap + f_val)
            
            # Write the whole vector as one line
            out.write("".join(row_vals_str) + "\n")

    print(f"Successfully created '{output_filename}'")

# --- EXECUTION ---
# Input: "upperhessian" (no extension)
# Output: "upperhessian.vwmatrix"
process_eigen_data('upperhessian', 'upperhessian.vwmatrix')
