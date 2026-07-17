#!/bin/bash
cd /Users/tommy/Downloads/mayi/PheroViz-c2-closure-integration-terra/nature_download

while true; do
    count=$(find outputs/nature_content -type f \( -name "fig_*.png" -o -name "fig_*.jpg" \) | wc -l)
    count=$((count + 0))
    if [ "$count" -ge 190 ]; then
        echo "[$(date)] Reached target of $count images (>=190). Stopping loop." >> /tmp/copilot-daemon-fixed.log
        break
    fi
    echo "[$(date)] Current count: $count. Starting/Restarting miner..." >> /tmp/copilot-daemon-fixed.log
    python3 nature_all_in_one.py auto --min-panels 5 --max-per-keyword 0 --max-articles 0 --stream --stream-workers 16 --require-cc-by >> /tmp/copilot-daemon-fixed.log 2>&1
    
    echo "[$(date)] Miner exited with code $?. Sleeping for 10s before restart..." >> /tmp/copilot-daemon-fixed.log
    sleep 10
done
