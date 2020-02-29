import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--T0', type=int, default=5000)
parser.add_argument('--numStages', type=int, default=10000)
parser.add_argument('--local_batchsize', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1) # initial learning rate
parser.add_argument('--gamma', type=float, default=2000)
parser.add_argument('--test_freq', type=int, default=800)
parser.add_argument('--test_batchsize', type=int, default=32)
parser.add_argument('--test_batches', type=int, default=100) # total used in testing" test_batchsize * test_batches
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--I', type=int, default=2)
parser.add_argument('--split_index', type=int, default=4) # split the classes, labels less than or equal to this are defined to be negative classes; 
                                                          #                    otherwise are defined to be positive classes
# parser.add_argument('--eval_alpha_local_batches', type=int, default=10)
parser.add_argument('--numGPU', type=int, default=1)
parser.add_argument('--total_iter', type=int, default=2000)
parser.add_argument('--neg_keep_ratio', type=float, default=1)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--master_addr', type=str)
parser.add_argument('--test_ratio', type=float, default=0.0001)

para = parser.parse_args()
