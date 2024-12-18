import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

# Define the block size
BLK_H = 8
BLK_W = 8

plot_graph_stat = True  # Set to True if you want to plot the sparsity patterns

def visualize_sparsity(sparse_matrix, title="Sparsity Pattern"):
    plt.figure(figsize=(10, 10))
    plt.spy(sparse_matrix, markersize=1)
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.savefig(title + ".png")
    plt.show()

# Replace with your datasets
graph_set = ["cant", "conf5_4-8x8-10", "consph", "cop20k_A", "dc2", "mip1", "pdb1HYS", "rma10", "shipsec1"]
num_graph = len(graph_set)
df_stats = np.zeros([num_graph, 14])

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
    print("Dataset:", dataset, "Num Rows:", num_rows, "Num NNZ:", num_nnz)
    df_stats[i, 0] = num_rows
    df_stats[i, 1] = num_nnz
    df_stats[i, 2] = num_nnz / (num_rows * num_rows)
    # Process data
    column_index = A_csr.indices  # Column indices
    row_pointers = A_csr.indptr   # Row pointers
    num_row_windows = (num_rows + BLK_H - 1) // BLK_H
    num_nnz = len(column_index)
    nBlockPerRowWindow = []
    nnzPerBlock = []
    for row_window_start in range(0, num_rows, BLK_H):
        row_window_end = min(row_window_start + BLK_H, num_rows)
        rows_in_window = row_window_end - row_window_start
        nnz_in_window = row_pointers[row_window_end] - row_pointers[row_window_start]
        col_indices_in_window = column_index[row_pointers[row_window_start]:row_pointers[row_window_end]]
        col_blocks = np.unique(col_indices_in_window // BLK_W)
        n_blocks = len(col_blocks)
        nBlockPerRowWindow.append(n_blocks)
        for col_block in col_blocks:
            col_block_start = col_block * BLK_W
            col_block_end = col_block_start + BLK_W
            nnz_in_block = 0
            for row in range(row_window_start, row_window_end):
                row_start = row_pointers[row]
                row_end = row_pointers[row + 1]
                row_cols = column_index[row_start:row_end]
                nnz_in_block += ((row_cols >= col_block_start) & (row_cols < col_block_end)).sum()
            nnzPerBlock.append(nnz_in_block)
    nBlockPerRowWindow = np.array(nBlockPerRowWindow)
    nnzPerBlock = np.array(nnzPerBlock)
    block_count = len(nnzPerBlock)
    df_stats[i, 3] = block_count
    # Compute statistics
    avg_nnz_per_block = nnzPerBlock.mean()
    avg_block_per_row_window = nBlockPerRowWindow.mean()
    variance_nnz_per_block = nnzPerBlock.var()
    variance_block_per_row_window = nBlockPerRowWindow.var()
    median_nnz_per_block = np.median(nnzPerBlock)
    first_quartile_nnz_per_block = np.percentile(nnzPerBlock, 25)
    third_quartile_nnz_per_block = np.percentile(nnzPerBlock, 75)
    median_block_per_row_window = np.median(nBlockPerRowWindow)
    first_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 25)
    third_quartile_block_per_row_window = np.percentile(nBlockPerRowWindow, 75)
    print("Average NNZ per Block:", avg_nnz_per_block)
    print("Variance NNZ per Block:", variance_nnz_per_block)
    print("Median NNZ per Block:", median_nnz_per_block)
    print("First Quartile NNZ per Block:", first_quartile_nnz_per_block)
    print("Third Quartile NNZ per Block:", third_quartile_nnz_per_block)
    print("Average Blocks per Row Window:", avg_block_per_row_window)
    print("Variance Blocks per Row Window:", variance_block_per_row_window)
    print("Median Blocks per Row Window:", median_block_per_row_window)
    print("First Quartile Blocks per Row Window:", first_quartile_block_per_row_window)
    print("Third Quartile Blocks per Row Window:", third_quartile_block_per_row_window)
    df_stats[i, 4] = avg_nnz_per_block
    df_stats[i, 5] = variance_nnz_per_block
    df_stats[i, 6] = median_nnz_per_block
    df_stats[i, 7] = first_quartile_nnz_per_block
    df_stats[i, 8] = third_quartile_nnz_per_block
    df_stats[i, 9] = avg_block_per_row_window
    df_stats[i, 10] = variance_block_per_row_window
    df_stats[i, 11] = median_block_per_row_window
    df_stats[i, 12] = first_quartile_block_per_row_window
    df_stats[i, 13] = third_quartile_block_per_row_window
    if plot_graph_stat:
        visualize_sparsity(A_csr[:1000, :1000], dataset + "_A[:1k, :1k]")
        visualize_sparsity(A_csr, dataset + "_full_A")
    print("================================================================")

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
