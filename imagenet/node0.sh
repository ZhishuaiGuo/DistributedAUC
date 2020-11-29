#!/bin/bash
python3 -m torch.distributed.launch \
               --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr="172.29.28.221" --master_port=2500 \
               main.py --master_addr="172.29.28.221" --T0=5000 --gamma=2000 --lr=0.1 --I=64 --local_batchsize=32 \
               --neg_keep_ratio=0.4 --total_iter=40000 --split_index=499 --test_ratio=0.01
sleep 15
