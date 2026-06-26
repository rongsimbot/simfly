#!/bin/bash
cd /tmp/simfly_web/phase17
VENV=/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3
SAVE=/tmp/simfly_web/phase17/models

JOINTS=( coxa_T2_left coxa_abduct_T2_left coxa_twist_T2_left femur_T2_left femur_twist_T2_left tibia_T2_left )

echo "=== T2 LEFT SEQUENCE: $(date) ===" | tee /tmp/simfly_web/phase17/sequence_t2_left.log
for joint in "${JOINTS[@]}"; do
  echo "  Training $joint..." | tee -a /tmp/simfly_web/phase17/sequence_t2_left.log
  $VENV quick_train.py --joint $joint --episodes 60 --steps 200 --save-dir $SAVE     > train_${joint}.log 2> train_${joint}.err
  best=$(grep 'Best:' train_${joint}.log 2>/dev/null | tail -1)
  echo "  $best" | tee -a /tmp/simfly_web/phase17/sequence_t2_left.log
done
echo "=== T2 LEFT COMPLETE: $(date) ===" | tee -a /tmp/simfly_web/phase17/sequence_t2_left.log
