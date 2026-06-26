#!/bin/bash
cd /tmp/simfly_web/phase17
VENV=/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3
SAVE=/tmp/simfly_web/phase17/models

JOINTS=(
  tibia_T1_left
  coxa_abduct_T1_right
  coxa_twist_T1_right
  femur_T1_right
  femur_twist_T1_right
  tibia_T1_right
)

echo "Starting batch at $(date)" | tee /tmp/simfly_web/phase17/sequence.log
for joint in "${JOINTS[@]}"; do
  echo "" | tee -a /tmp/simfly_web/phase17/sequence.log
  echo "=== TRAINING $joint ===" | tee -a /tmp/simfly_web/phase17/sequence.log
  $VENV quick_train.py --joint $joint --episodes 60 --steps 200 --save-dir $SAVE     > train_${joint}.log 2> train_${joint}.err
  best=$(grep 'Best:' train_${joint}.log 2>/dev/null | tail -1)
  avg=$(grep 'Final avg10' train_${joint}.log 2>/dev/null | tail -1)
  echo "  $best | $avg" | tee -a /tmp/simfly_web/phase17/sequence.log
done
echo "" | tee -a /tmp/simfly_web/phase17/sequence.log
echo "=== SEQUENCE COMPLETE: $(date) ===" | tee -a /tmp/simfly_web/phase17/sequence.log
