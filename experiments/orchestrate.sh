#!/bin/bash
cd /home/j-i15a204/tre
log(){ echo "[ORCH $(date +%H:%M:%S)] $1"; }
count_train(){ echo $(( $(ls features/train/fake 2>/dev/null | wc -l) + $(ls features/train/real 2>/dev/null | wc -l) )); }

log "waiting for 60000 train features"
while [ $(count_train) -lt 60000 ]; do sleep 120; done
log "train features complete -> main training (GPU2, concurrent with test extraction)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./venv/bin/python train_eval.py --epochs 20 --batch 16 --results results_partial.json > train_main.log 2>&1
log "main training done"

log "waiting for all 4 extraction shards"
while [ $(grep -l ALL_DONE extract_gpu*.log 2>/dev/null | wc -l) -lt 4 ]; do sleep 300; done
log "extraction complete -> final repro eval"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./venv/bin/python train_eval.py --eval-only --results results.json > eval_repro.log 2>&1
log "repro results -> results.json"

log "launching fresh extraction on 4 GPUs"
for k in 0 1 2 3; do
  bash driver_gpu_fresh.sh $k > extract_fresh_gpu$k.log 2>&1 &
done
wait
log "fresh extraction done -> fresh training"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./venv/bin/python train_eval.py --features /home/j-i15a204/tre/features_fresh --epochs 20 --batch 16 --ckpt weights/ours_tre_fresh.pt --results results_fresh.json > train_fresh.log 2>&1
log "ALL_PHASES_DONE"
