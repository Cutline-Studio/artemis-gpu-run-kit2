#!/bin/bash
DST=/workspace/a567-dense/checkpoints/gpu-run2-20261025; END=1793002789; mkdir -p $DST
log(){ echo "$(date '+%F %T %Z') $*" >> /workspace/gpu-run-kit2/harvester2.log; }
log start
while [ $(date +%s) -lt $END ]; do
  ssh -F /workspace/gpu-run-kit2/ssh_cfg -o ConnectTimeout=20 gpu 'test -f /workspace/run/out/m94/DONE' 2>/dev/null && { log DONE-seen; break; }
  sleep 60
done
for i in 1 2 3; do
  timeout 900 rsync -a -e "ssh -F /workspace/gpu-run-kit2/ssh_cfg" --exclude '*.tmp' gpu:/workspace/run/out/ gpu:/workspace/run/data/fetch_progress.json gpu:/workspace/run/base_token_counts.json gpu:/workspace/run/train_stdout.log gpu:/workspace/run/fetch_stdout.log $DST/ >> /workspace/gpu-run-kit2/harvester2.log 2>&1; rc=$?
  log "rsync rc=$rc"; [ $rc -eq 0 ] && break; sleep 10
done
for i in 1 2 3 4 5; do /workspace/vast-venv/bin/vastai destroy instance 57753825 -y --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit2/harvester2.log 2>&1 && { log destroyed; break; }; sleep 15; done
sleep 20; /workspace/vast-venv/bin/vastai show instance 57753825 --raw --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit2/harvester2.log 2>&1
log end
