#!/usr/bin/env bash

# CONFIG='/workspace/FlashOCC/projects/configs/dual_occ/dualocc-r50-sat2.py'
# CHECKPOINT='/workspace/FlashOCC/work_dirs/dualocc-r50-sat-onlysat2/epoch_24_ema.pth'
CONFIG='/workspace/FlashOCC/projects/configs/dual_occ/dualocc-r50-sat.py'
CHECKPOINT='/workspace/FlashOCC/work_dirs/dualocc-r50-sat-final2-nod-sa/epoch_24_ema.pth'
# python tools/vis_occ.py --config /workspace/FlashOCC/projects/configs/flashocc/flashocc-r50.py --weights /workspace/FlashOCC/flashocc-r50-256x704.pth --viz-dir vis/flashocc

GPUS=1
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
    ${@:8}
    # --eval-options show_dir=vis/dualocc-r50-sat-final2 \
    # --eval-options show_dir=vis/sat \
    # --eval mAP \   
