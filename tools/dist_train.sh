#!/usr/bin/env bash

# CONFIG=$1  TORCH_DISTRIBUTED_DEBUG=DETAIL 
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --seed 0 \
    --auto-resume \
    --launcher pytorch ${@:3} > dualocc-r50-sat-final2-nod-sa.log 2>&1 &
