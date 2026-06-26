#!/bin/bash
cd /tmp/simfly_web/phase17
VENV=/home/simllm/simrobotics-storage/research/flywire/virtual-fly/venv/bin/python3
SAVE=/tmp/simfly_web/phase17/models

JOINTS=( coxa_T3_right coxa_abduct_T3_right coxa_twist_T3_right femur_T3_right femur_twist_T3_right tibia_T3_right )

echo "=== T3 RIGHT SEQUENCE: $(date) ===" | tee /tmp/simfly_web/phase17/sequence_t3_right.log
for joint in "${JOINTS[@]}"; do
  echo "  Training $joint..." | tee -a /tmp/simfly_web/phase17/sequence_t3_right.log
  $VENV quick_train.py --joint $joint --episodes 60 --steps 200 --save-dir $SAVE \
    > train_${joint}.log 2> train_${joint}.err
  best=$(grep "Best:" train_${joint}.log 2>/dev/null | tail -1)
  echo "  $best" | tee -a /tmp/simfly_web/phase17/sequence_t3_right.log
done
echo "=== T3 RIGHT COMPLETE: $(date) ===" | tee -a /tmp/simfly_web/phase17/sequence_t3_right.log
echo "ALL_36_COMPLETE" > /tmp/simfly_web/phase17/ALL_DONE
