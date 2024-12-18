import os.path as osp
import argparse
import torch

BLK_H = 16
BLK_W = 8
import DTCSpMM
from dataset import *

ExecutionPlan = {
  # reorderd
#  "YeastH.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#  "OVCAR-8H.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#  "Yeast.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#  "DD.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float4", "split"]},
#  "web-BerkStan.reorder": {128: [False, "float2", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "nonsplit"]},
#  "reddit.reorder": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#  "ddi.reorder": {128: [True, "float", "nonsplit"], 256: [True, "float", "nonsplit"], 512: [True, "float4", "split"]},
#  "protein.reorder": {128: [False, "float4", "split"], 256: [False, "float4", "split"], 512: [False, "float4", "split"]},
  
  # origin
#   "cop20k_A": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "cant": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "consph": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "pdb1HYS": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "dc2": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "mip1": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "conf5_4-8x8-10": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
   "rma10": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "shipsec1": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},

#   "YeastH": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "OVCAR-8H": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "Yeast": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "DD": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float4", "split"]},
#   "web-BerkStan": {128: [False, "float2", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "nonsplit"]},
#   "reddit": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"]},
#   "ddi": {128: [True, "float", "nonsplit"], 256: [True, "float", "nonsplit"], 512: [True, "float4", "split"]},
#   "protein": {128: [False, "float4", "split"], 256: [False, "float4", "split"], 512: [False, "float4", "split"]},
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='YeastH', help="dataset")
parser.add_argument("--feat_size", type=int, default=128, help="embedding size")
args = parser.parse_args()
print(args)

## Load matrix from files.
dataset = args.dataset
dset_name = dataset
# Set your own path to the dataset.
path = osp.join("/pub/hsud8/DTC-SpMM_ASPLOS24/DTC-SpMM/dtc_datasets/", dataset + ".npz") #4090
matrix = DTC_dataset(path)
num_rows = matrix.num_nodes
num_nnz = matrix.num_edges
print("NUM_ROW, NNZ: ", num_rows, " " , num_nnz)
column_index =  matrix.column_index 
row_pointers =  matrix.row_pointers 
# Process data.
num_row_windows = (num_rows + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
column_index_ori  = column_index.cuda()
row_pointers_ori = row_pointers.cuda()

blockPartition_cuda  = blockPartition.cuda()
edgeToColumn_cuda = edgeToColumn.cuda()
edgeToRow_cuda  = edgeToRow.cuda()

# Optimize GPU.
RowWindowOffset, TCblockRowid,\
      TCblocktileId, TCblockoffset, SparseAToXindex,\
        block_count = DTCSpMM.preprocess_gpu(column_index_ori, row_pointers_ori, num_rows, BLK_H, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)

# Run tests.
print("running DTC_SPMM")
print("args.feat_size =", args.feat_size)
X = torch.ones((num_rows, args.feat_size)).cuda()
# Run test.
with open("./DTCSpMM_exe_time_and_throughput.csv", 'a') as f:
    f.write(dset_name + "," + str(args.feat_size)  + ",")
balance_choice = ExecutionPlan[dset_name][args.feat_size][0]
exeplan = ExecutionPlan[dset_name][args.feat_size][1] + "_" + ExecutionPlan[dset_name][args.feat_size][2]
if balance_choice == False:
    X_out = DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, num_nnz, exeplan)[0]
    print("X_out:", X_out)
    print("X_out shape:", X_out.shape)
else:
    X_out = DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, exeplan)[0]
    print("X_out:", X_out)
    print("X_out shape:", X_out.shape)
with open("./DTCSpMM_exe_time_and_throughput.csv", 'a') as f:
    f.write("\n")
