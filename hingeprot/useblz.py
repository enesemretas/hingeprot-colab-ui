import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import eigsh
import time

def solve_sparse_eigenproblem(matrix_file):
    print(f"Reading matrix from {matrix_file}...")
    
    try:
        with open(matrix_file, 'r') as f:
            # Read total number of non-zero elements
            line = f.readline().split()
            if not line:
                return
            na = int(line[0])
            
            rows = []
            cols = []
            data = []
            
            # Read (irn, jcn, sa) triples
            for _ in range(na):
                line = f.readline().split()
                if not line:
                    break
                # Fortran is 1-indexed, Python is 0-indexed
                r, c, val = int(line[0]) - 1, int(line[1]) - 1, float(line[2])
                rows.append(r)
                cols.append(c)
                data.append(val)
                
                # If only upper triangular is stored, we must mirror it for SciPy
                if r != c:
                    rows.append(c)
                    cols.append(r)
                    data.append(val)

        # Determine matrix dimension N
        n = max(max(rows), max(cols)) + 1
        print(f"Matrix Dimension N: {n}, Non-zeros: {len(data)}")

        # Construct Sparse Matrix (CSC format is best for solvers)
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
        
    except FileNotFoundError:
        print(f"Error: File {matrix_file} not found.")
        return

    # Configuration (Matching the Fortran istor/rstor parameters)
    n_eig_required = 36  # istor(3)
    sigma = 0.0          # rstor(1) - The shift value
    
    print(f"Solving for {n_eig_required} eigenvalues near sigma={sigma}...")
    start_time = time.time()

    # Solve Ax = lambda Bx (B is identity by default in the Fortran code)
    # 'LM' with sigma finds eigenvalues closest to sigma (Shift-and-Invert)
    eigenvalues, eigenvectors = eigsh(A, k=n_eig_required, sigma=sigma, which='LM')

    end_time = time.time()
    cpu_used = (end_time - start_time) / 60.0
    print(f"Total CPU used: {cpu_used:.4f} minutes")

    # Save results to .vwmatrixd (Matching Fortran output)
    output_file = matrix_file.split('.')[0] + ".vwmatrixd"
    with open(output_file, 'w') as f:
        f.write(f"{len(eigenvalues)} {n}\n")
        
        # Write eigenvalues
        for ev in eigenvalues:
            # Fortran EIG(i,2) usually stores imaginary part; for symmetric A, it's 0
            f.write(f"{ev:20.12e} {0.0:20.12e}\n")
            
        # Write eigenvectors (flattened as per the Fortran write statement)
        # Note: Fortran writes column-major (all elements of v1, then all of v2...)
        for j in range(len(eigenvalues)):
            for i in range(n):
                f.write(f"{eigenvectors[i, j]:20.12e}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    solve_sparse_eigenproblem('upperhessian')
