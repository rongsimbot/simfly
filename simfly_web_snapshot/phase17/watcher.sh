#!/bin/bash
# Watcher: kicks off next batch when current one finishes, runs all remaining
cd /tmp/simfly_web/phase17

T2_RIGHT_DONE="/tmp/simfly_web/phase17/.t2_right_done"
T3_LEFT_DONE="/tmp/simfly_web/phase17/.t3_left_done"
T3_RIGHT_DONE="/tmp/simfly_web/phase17/.t3_right_done"

# Wait for T2 right to finish (check every 60s)
while pgrep -f "run_t2_right.sh" > /dev/null 2>&1; do
  sleep 60
done
echo "T2_RIGHT_DONE_AT=$(date)" > "$T2_RIGHT_DONE"

# Kick off T3 left
echo "=== STARTING T3 LEFT: $(date) ==="
bash run_t3_left.sh
echo "T3_LEFT_DONE_AT=$(date)" > "$T3_LEFT_DONE"

# Kick off T3 right
echo "=== STARTING T3 RIGHT: $(date) ==="
bash run_t3_right.sh
echo "T3_RIGHT_DONE_AT=$(date)" > "$T3_RIGHT_DONE"
