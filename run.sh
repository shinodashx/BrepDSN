#!/bin/bash

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIME_STR=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/run_graphgym_${TIME_STR}.log"

export CUDA_VISIBLE_DEVICES=0
nohup python graphgym/main.py --cfg graphgym/configs/2025_0625_Brep_CC3D_Full_Enhanced_SL_Final.yaml seed 4 > "$LOG_FILE" 2>&1 &
