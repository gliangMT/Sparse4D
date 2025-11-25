#!/usr/bin/env bash

# --- Detect hardware type ---
if command -v nvidia-smi &> /dev/null; then
    HW="cuda"
elif command -v mthreads-gmi &> /dev/null; then
    HW="musa"
else
    HW="unknown"
fi

echo "[INFO] Detected hardware: $HW"

CONFIG=$1
GPUS=$2
PORT=${PORT:-28650}

# --- Select train script based on hardware ---
if [ "$HW" == "musa" ]; then
    TRAIN_SCRIPT="$(dirname "$0")/train_musa.py"
else
    TRAIN_SCRIPT="$(dirname "$0")/train.py"   # default CUDA
fi

echo "[INFO] Using training script: $TRAIN_SCRIPT"

# --- Launch distributed training ---
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# Previous torch.distributed.launch is deprecated
# python3 -m torch.distributed.launch \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $TRAIN_SCRIPT $CONFIG --launcher pytorch ${@:3}

# New recommended way using torchrun
torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $TRAIN_SCRIPT \
    $CONFIG \
    --launcher pytorch \
    ${@:3}