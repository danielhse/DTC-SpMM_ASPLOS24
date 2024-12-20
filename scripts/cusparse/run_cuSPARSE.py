import os.path as osp
import argparse
import torch

import DTCSpMM
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='YeastH', help="dataset")
args = parser.parse_args()
print(args)

## Load matrix from files.
dataset = args.dataset

# Set your own path to the datasets.
path = osp.join("/pub/hsud8/DTC-SpMM_ASPLOS24/DTC-SpMM/dtc_datasets/", args.dataset + ".npz") #4090
matrix = DTC_dataset(path)
num_rows = matrix.num_nodes
num_nnz = matrix.num_edges
print("M, E: ", num_rows, " " , num_nnz)
column_index =  matrix.column_index 
row_pointers = matrix.row_pointers

column_index_ori  = column_index.cuda()
row_pointers_ori = row_pointers.cuda()

# Run tests.
for feat_size in [128, 256, 512]:
    print("feat_size =", feat_size)
    X = torch.ones((num_rows, feat_size)).cuda()
    # Run test.
    with open("./cuSPARSE_exe_time_and_throughput.csv", 'a') as f:
        f.write(dataset + "," + str(feat_size)  + ",")
    out = DTCSpMM.run_cuSPARSE(X, row_pointers_ori, column_index_ori, num_rows, -1)
    with open("./cuSPARSE_exe_time_and_throughput.csv", 'a') as f:
        f.write("\n")

