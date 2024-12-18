import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix

# Define file paths
mtx_file = 'rma10.mtx'
npz_file = 'rma10.npz'

# Step 1: Read the .mtx file and extract a submatrix
print("Reading .mtx file...")
sparse_matrix_mtx = mmread(mtx_file).tocsr()
row_start, row_end = 25, 79  # Adjust as needed
col_start, col_end = 25, 79  # Adjust as needed
submatrix_mtx = sparse_matrix_mtx[row_start:row_end, col_start:col_end]
dense_submatrix_mtx = submatrix_mtx.toarray()

# Step 2: Load the .npz file and extract the same submatrix
print("Loading .npz file...")
npz_data = np.load(npz_file)

# Verify available keys in the .npz file
print("Keys in .npz file:", npz_data.files)

# Retrieve the shape
if 'shape' in npz_data:
    num_rows, num_cols = npz_data['shape']
    print(f"Shape from .npz file: {num_rows} rows, {num_cols} columns")
else:
    raise KeyError("The key 'shape' is not found in the .npz file. Please check the conversion script.")

# Extract source, destination indices, and edge weights
src_li = npz_data['src_li']
dst_li = npz_data['dst_li']
edge_weights = npz_data['edge_weights']

# Reconstruct the sparse matrix using edge weights and original shape
print("Reconstructing sparse matrix from .npz data...")
sparse_matrix_npz = coo_matrix(
    (edge_weights, (src_li, dst_li)),
    shape=(num_rows, num_cols)
).tocsr()

submatrix_npz = sparse_matrix_npz[row_start:row_end, col_start:col_end]
dense_submatrix_npz = submatrix_npz.toarray()

# Ensure both submatrices have the same data type
print("Data type of submatrix_mtx:", dense_submatrix_mtx.dtype)
print("Data type of submatrix_npz before conversion:", dense_submatrix_npz.dtype)

if dense_submatrix_mtx.dtype != dense_submatrix_npz.dtype:
    print("Converting submatrix_npz to match dtype of submatrix_mtx...")
    dense_submatrix_npz = dense_submatrix_npz.astype(dense_submatrix_mtx.dtype)
    print("Data type of submatrix_npz after conversion:", dense_submatrix_npz.dtype)
else:
    print("Data types match.")

# Step 3: Compare the two submatrices
print("Comparing submatrices...")
difference = dense_submatrix_mtx - dense_submatrix_npz
max_diff = np.max(np.abs(difference))
has_nans = np.isnan(difference).any()
has_infs = np.isinf(difference).any()
are_equal = np.allclose(dense_submatrix_mtx, dense_submatrix_npz)

print("Are the submatrices equal?", are_equal)
print("Max absolute difference:", max_diff)
print("Any NaNs in difference:", has_nans)
print("Any Infs in difference:", has_infs)

# Optional: Display the submatrices
print("\nSubmatrix from .mtx file:")
print(dense_submatrix_mtx)

print("\nSubmatrix from .npz file:")
print(dense_submatrix_npz)

# Optional: Display differences if not equal
if not are_equal:
    print("\nDifference between submatrices:")
    print(difference)
