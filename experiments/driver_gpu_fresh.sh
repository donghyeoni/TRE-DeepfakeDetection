#!/bin/bash
cd /home/j-i15a204/tre
K=$1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$K
log() { echo "[FRESH-GPU$K $(date +%H:%M:%S)] $1"; }
log "start fresh shard $K"
./venv/bin/python extract_tre.py --list data/lists/train.txt --out features_fresh/train --batch 48 --shard $K --nshards 4 --fresh
log "train shard done"
for gen in sdv4 sdv5 adm biggan glide midjourney vqdm wukong; do
  log "test $gen"
  ./venv/bin/python extract_tre.py --list data/lists/test_$gen.txt --out features_fresh/test/$gen --batch 48 --shard $K --nshards 4 --fresh
done
log "FRESH_SHARD_DONE"
