#!/usr/bin/env bash

CONFIG=${1:?"usage: $0 CONFIG CHECKPOINT [GPUS] [extra test args...]"}
CHECKPOINT=${2:?"usage: $0 CONFIG CHECKPOINT [GPUS] [extra test args...]"}
GPUS=${3:-1}
shift 3

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29509}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.9"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval mAP \
    --launcher pytorch \
    "$@"
    # --eval-options show_dir=vis/dualocc-r50-sat-final2 \
    # --eval-options show_dir=vis/sat \
    # --eval mAP \   
