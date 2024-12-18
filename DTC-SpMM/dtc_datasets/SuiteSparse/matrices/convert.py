#!/usr/bin/env python3

import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix
import argparse

def convert_mtx_to_npz(mtx_file, npz_file, undirected=False, remove_self_loops=False):
    print(f"Reading the .mtx file '{mtx_file}'...")
    sparse_matrix = mmread(mtx_file)

    # Handle dense and sparse matrices
    if isinstance(sparse_matrix, np.ndarray):
        print("The matrix is dense. Converting to COO sparse format...")
        sparse_matrix = coo_matrix(sparse_matrix)
    else:
        print("The matrix is sparse. Converting to COO format...")
        sparse_matrix = sparse_matrix.tocoo()

    # Adjust for one-based indexing
    if sparse_matrix.row.min() == 1 or sparse_matrix.col.min() == 1:
        print("Adjusting indices to be zero-based...")
        sparse_matrix.row -= 1
        sparse_matrix.col -= 1

    src_li = sparse_matrix.row
    dst_li = sparse_matrix.col
    edge_weights = sparse_matrix.data

    # Make undirected if needed
    if undirected:
        print("Converting to an undirected graph by adding reverse edges...")
        src_li = np.concatenate([src_li, dst_li])
        dst_li = np.concatenate([dst_li, src_li[:len(dst_li)]])
        edge_weights = np.concatenate([edge_weights, edge_weights])

    # Remove self-loops if needed
    if remove_self_loops:
        print("Removing self-loops...")
        mask = src_li != dst_li
        src_li = src_li[mask]
        dst_li = dst_li[mask]
        edge_weights = edge_weights[mask]

    # Remove duplicate edges by summing weights
    print("Removing duplicate edges...")
    edges = np.vstack((src_li, dst_li)).T
    unique_edges, inverse_indices = np.unique(edges, axis=0, return_inverse=True)
    summed_weights = np.zeros(len(unique_edges), dtype=edge_weights.dtype)
    np.add.at(summed_weights, inverse_indices, edge_weights)
    src_li, dst_li, edge_weights = unique_edges[:,0], unique_edges[:,1], summed_weights

    num_rows, num_cols = sparse_matrix.shape
    num_nodes = num_rows  # Assuming num_nodes corresponds to rows

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges after processing: {len(src_li)}")

    # Save to .npz with 'num_nodes'
    print(f"Saving to .npz file '{npz_file}'...")
    np.savez(
        npz_file,
        src_li=src_li,
        dst_li=dst_li,
        edge_weights=edge_weights,
        num_nodes=num_nodes,
        shape=(num_rows, num_cols)
    )
    print("Conversion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a .mtx file to a .npz file.')
    parser.add_argument('mtx_file', type=str, help='Path to the input .mtx file.')
    parser.add_argument('npz_file', type=str, help='Path to the output .npz file.')
    parser.add_argument('--undirected', action='store_true', help='Convert the graph to undirected by adding reverse edges.')
    parser.add_argument('--remove_self_loops', action='store_true', help='Remove self-loops from the graph.')

    args = parser.parse_args()

    convert_mtx_to_npz(
        mtx_file=args.mtx_file,
        npz_file=args.npz_file,
        undirected=args.undirected,
        remove_self_loops=args.remove_self_loops
    )
