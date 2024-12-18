import os.path as osp
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# Define the block size
BLK_H = 8
BLK_W = 8

plot_graph_stat = True  # Set to True to generate and save the sparsity patterns

# List of your datasets
graph_set = ["cant", "conf5_4-8x8-10", "consph", "cop20k_A", "dc2", "mip1", "pdb1HYS", "rma10", "shipsec1"]
num_graph = len(graph_set)
df_stats = np.zeros([num_graph, 14])

# Create a figure with subplots arranged in columns
fig, axes = plt.subplots(nrows=2, ncols=num_graph, figsize=(num_graph * 5, 12))

for i, dataset in enumerate(graph_set):
    # Set the path to your dataset
    path = osp.join("/pub/hsud8/DTC-SpMM_ASPLOS24/DTC-SpMM/dtc_datasets/patterns", dataset + ".npz")
    # Load the .npz file
    npzfile = np.load(path)
    print("Keys in the npz file for dataset", dataset, ":", npzfile.files)

    # Check for edge list keys
    if set(['src_li', 'dst_li', 'edge_weights', 'num_nodes']).issubset(npzfile.files):
        src_li = npzfile['src_li']
        dst_li = npzfile['dst_li']
        edge_weights = npzfile['edge_weights']
        num_nodes = npzfile['num_nodes'].item()  # Ensure num_nodes is an integer

        # Create a COO sparse matrix from the edge lists
        A_coo = coo_matrix((edge_weights, (src_li, dst_li)), shape=(num_nodes, num_nodes))
        # Convert to CSR format for efficient row slicing
        A_csr = A_coo.tocsr()
    else:
        raise ValueError(f"Unexpected data format in npz file {dataset}.npz")

    num_rows = A_csr.shape[0]
    num_nnz = A_csr.nnz
    nnz_percentage = num_nnz / (num_rows * num_rows)
    print("Dataset:", dataset, "Num Rows:", num_rows, "Num NNZ:", num_nnz)
    df_stats[i, 0] = num_rows
    df_stats[i, 1] = num_nnz
    df_stats[i, 2] = nnz_percentage
    # Process data (omitted for brevity; same as before)
    # ...

    # Plotting
    if plot_graph_stat:
        # Plot the first 1000x1000 submatrix
        ax1 = axes[0, i]
        ax1.spy(A_csr[:1000, :1000], markersize=1)
        ax1.set_title(f"{dataset} A[0:1k, 0:1k]", fontsize=10)
        if i == 0:
            ax1.set_ylabel("Row Index")
        ax1.set_xlabel("Column Index")

        # Plot the full matrix
        ax2 = axes[1, i]
        ax2.spy(A_csr, markersize=1)
        ax2.set_title(f"{dataset} Full A", fontsize=10)
        if i == 0:
            ax2.set_ylabel("Row Index")
        ax2.set_xlabel("Column Index")

        # Add dataset info above the plots
        info_text = f"Dataset: {dataset}\nNum Rows: {num_rows}\nNNZ Ratio: {nnz_percentage:.2e}"
        # Place the text above the first subplot of the column
        ax1.text(0.5, 1.2, info_text, ha='center', va='bottom', transform=ax1.transAxes, fontsize=12)
    print("================================================================")

# Adjust layout and save the combined figure
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.4)
plt.savefig(f"{BLK_H}x{BLK_W}_graph_sparsity_patterns.png")
plt.close()

# Save the statistics to a CSV file (same as before)
columns = [
    "num_rows", "num_nnz", "nnz_percentage", "block_count", "avg_nnz_per_block",
    "variance_nnz_per_block", "median_nnz_per_block", "first_quartile_nnz_per_block",
    "third_quartile_nnz_per_block", "avg_block_per_row_window", "variance_block_per_row_window",
    "median_block_per_row_window", "first_quartile_block_per_row_window", "third_quartile_block_per_row_window"
]
df_stats = pd.DataFrame(df_stats, columns=columns)
df_stats.insert(0, 'graph', graph_set)
df_stats.to_csv(f"{BLK_H}x{BLK_W}_graph_stat.csv", index=False)
