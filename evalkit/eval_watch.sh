#!/bin/bash
# waits for training DONE, runs the GPU comparison eval, then marks EVAL_DONE (harvester waits for it)
cd /workspace/run
while [ ! -f out/m94/DONE ]; do sleep 30; done
timeout 1500 python /workspace/eval_gpu.py > out/m94/eval_compare.log 2>&1; echo "rc=$?" >> out/m94/eval_compare.log
touch out/m94/EVAL_DONE
