#!/bin/bash
# usage: guards.sh IID RENT_EPOCH  -> starts detached local harvester (pull on DONE / at rent+5h50m, then destroy+verify)
# and detached hard-kill (destroy at rent+6h00m). Logs in /workspace/gpu-run-kit2/.
K=/workspace/gpu-run-kit2; IID=$1; R=$2; V=/workspace/vast-venv/bin/vastai
cat > $K/killer2.sh <<EOK
#!/bin/bash
END=$((R + 21600))
while [ \$(date +%s) -lt \$END ]; do sleep 20; [ -f $K/STOP_KILLER ] && exit 0; done
echo "\$(date) hard-kill firing" >> $K/killer2.log
for i in 1 2 3 4 5; do $V destroy instance $IID -y --api-key "\$VAST_API_KEY" >> $K/killer2.log 2>&1 && break; sleep 15; done
$V show instance $IID --raw --api-key "\$VAST_API_KEY" >> $K/killer2.log 2>&1
EOK
cat > $K/harvester2.sh <<EOH
#!/bin/bash
DST=/workspace/a567-dense/checkpoints/gpu-run2-20261025; END=$((R + 21000)); mkdir -p \$DST
log(){ echo "\$(date '+%F %T %Z') \$*" >> $K/harvester2.log; }
log start
while [ \$(date +%s) -lt \$END ]; do
  ssh -F $K/ssh_cfg -o ConnectTimeout=20 gpu 'test -f /workspace/run/out/m94/DONE' 2>/dev/null && { log DONE-seen; break; }
  sleep 60
done
for i in 1 2 3; do
  timeout 900 rsync -a -e "ssh -F $K/ssh_cfg" --exclude '*.tmp' gpu:/workspace/run/out/ gpu:/workspace/run/data/fetch_progress.json gpu:/workspace/run/base_token_counts.json gpu:/workspace/run/train_stdout.log gpu:/workspace/run/fetch_stdout.log \$DST/ >> $K/harvester2.log 2>&1; rc=\$?
  log "rsync rc=\$rc"; [ \$rc -eq 0 ] && break; sleep 10
done
for i in 1 2 3 4 5; do $V destroy instance $IID -y --api-key "\$VAST_API_KEY" >> $K/harvester2.log 2>&1 && { log destroyed; break; }; sleep 15; done
sleep 20; $V show instance $IID --raw --api-key "\$VAST_API_KEY" >> $K/harvester2.log 2>&1
log end
EOH
chmod +x $K/killer2.sh $K/harvester2.sh
cd $K && setsid nohup ./killer2.sh >/dev/null 2>&1 </dev/null & 
cd $K && setsid nohup ./harvester2.sh >/dev/null 2>&1 </dev/null &
sleep 1; pgrep -af "killer2|harvester2"
