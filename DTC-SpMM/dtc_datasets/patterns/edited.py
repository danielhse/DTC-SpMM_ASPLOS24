import os.path as osp
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Block size
BLK_H = 8
BLK_W = 8

plot_graph_stat = True  # Enable plotting

# List of datasets
graph_set = ["cant", "conf5_4-8x8-10", "consph", "cop20k_A", "dc2", "mip1", "pdb1HYS", "rma10", "shipsec1"]
num_graph = len(graph_set)
df_stats = np.zeros([num_graph, 14])

# Create figure with subplots arranged in columns
fig_height = 12  # Adjusted height
fig, axes = plt.subplots(nrows=2, ncols=num_graph, figsize=(num_graph * 5, fig_height))

for i, dataset in enumerate(graph_set):
    # Load dataset
    path = osp.join("/pub/hsud8/DTC-SpMM_ASPLOS24/DTC-SpMM/dtc_datasets/patterns", dataset + ".npz")
    npzfile = np.load(path)
    print(f"Processing dataset: {dataset}")

    # Build sparse matrix
    if {'src_li', 'dst_li', 'edge_weights', 'num_nodes'}.issubset(npzfile.files):
        src_li = npzfile['src_li']
        dst_li = npzfile['dst_li']
        edge_weights = npzfile['edge_weights']
        num_nodes = int(npzfile['num_nodes'])
        A_coo = coo_matrix((edge_weights, (src_li, dst_li)), shape=(num_nodes, num_nodes))
        A_csr = A_coo.tocsr()
    else:
        raise ValueError(f"Unexpected data format in {dataset}.npz")

    num_rows = A_csr.shape[0]
    num_nnz = A_csr.nnz
    nnz_percentage = num_nnz / (num_rows * num_rows)
    df_stats[i, :3] = [num_rows, num_nnz, nnz_percentage]

    # Plotting
    if plot_graph_stat:
        # Dataset name
        dataset_text = f"{dataset}"
        ax1 = axes[0, i]
        ax1.text(0.5, 1.35, dataset_text, ha='center', va='bottom', transform=ax1.transAxes, fontsize=16, fontweight='bold')

        # Num Rows and NNZ Ratio
        info_text = f"Num Rows: {num_rows}\nNNZ Ratio: {nnz_percentage:.2e}"
        ax1.text(0.5, 1.18, info_text, ha='center', va='bottom', transform=ax1.transAxes, fontsize=12)

        # Plot A[:1k, :1k]
        ax1.spy(A_csr[:1000, :1000], markersize=1)
        ax1.set_title(f"A[:1k, :1k]", fontsize=10)
        if i == 0:
            ax1.set_ylabel("Row Index")
#        ax1.set_xticks([])
#        ax1.set_yticks([])

        # Plot Full A
        ax2 = axes[1, i]
        ax2.spy(A_csr, markersize=1)
        ax2.set_title("Full A", fontsize=10)
        if i == 0:
            ax2.set_ylabel("Row Index")
#        ax2.set_xticks([])
#        ax2.set_yticks([])

    print("================================================================")

# Adjust layout
plt.tight_layout(h_pad=2.0)  # Decrease vertical space between plots
plt.subplots_adjust(top=0.85)  # Increase top margin for text
plt.savefig(f"{BLK_H}x{BLK_W}_graph_sparsity_patterns.png")
plt.close()

# Save the statistics to a CSV file
columns = [
    "num_rows", "num_nnz", "nnz_percentage", "block_count", "avg_nnz_per_block",
    "variance_nnz_per_block", "median_nnz_per_block", "first_quartile_nnz_per_block",
    "third_quartile_nnz_per_block", "avg_block_per_row_window", "variance_block_per_row_window",
    "median_block_per_row_window", "first_quartile_block_per_row_window", "third_quartile_block_per_row_window"
]
df_stats = pd.DataFrame(df_stats, columns=columns)
df_stats.insert(0, 'graph', graph_set)
df_stats.to_csv(f"{BLK_H}x{BLK_W}_graph_stat.csv", index=False)
