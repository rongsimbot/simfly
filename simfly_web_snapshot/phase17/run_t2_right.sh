#!/bin/bash
cd /tmp/simfly_web/phase17
VENV=/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3
SAVE=/tmp/simfly_web/phase17/models

JOINTS=( coxa_T2_right coxa_abduct_T2_right coxa_twist_T2_right femur_T2_right femur_twist_T2_right tibia_T2_right )

echo "=== T2 RIGHT SEQUENCE: $(date) ===" | tee /tmp/simfly_web/phase17/sequence_t2_right.log
for joint in "${JOINTS[@]}"; do
  echo "  Training $joint..." | tee -a /tmp/simfly_web/phase17/sequence_t2_right.log
  $VENV quick_train.py --joint $joint --episodes 60 --steps 200 --save-dir $SAVE \
    > train_${joint}.log 2> train_${joint}.err
  best=$(grep 'Best:' train_${joint}.log 2>/dev/null | tail -1)
  echo "  $best" | tee -a /tmp/simfly_web/phase17/sequence_t2_right.log
done
echo "=== T2 RIGHT COMPLETE: $(date) ===" | tee -a /tmp/simfly_web/phase17/sequence_t2_right.log
