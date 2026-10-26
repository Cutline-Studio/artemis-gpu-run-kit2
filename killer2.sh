#!/bin/bash
END=1793003389
while [ $(date +%s) -lt $END ]; do sleep 20; [ -f /workspace/gpu-run-kit2/STOP_KILLER ] && exit 0; done
echo "$(date) hard-kill firing" >> /workspace/gpu-run-kit2/killer2.log
for i in 1 2 3 4 5; do /workspace/vast-venv/bin/vastai destroy instance 57753825 -y --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit2/killer2.log 2>&1 && break; sleep 15; done
/workspace/vast-venv/bin/vastai show instance 57753825 --raw --api-key "$VAST_API_KEY" >> /workspace/gpu-run-kit2/killer2.log 2>&1
