#!/bin/bash
# 学習を再開するスクリプト（last.ptから再開）
# Usage: ./bin/resume_train.sh [output_log_file]

set -e

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="outputs/logs"
LOG_FILE="${1:-${LOG_DIR}/train_${TIMESTAMP}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "================================================================================"
echo "Resuming training session"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: $LOG_FILE"
echo "================================================================================"

if [ ! -f "outputs/checkpoints/last.pt" ]; then
    echo "No checkpoint found at outputs/checkpoints/last.pt"
    exit 1
fi

echo "Training output redirected to: $LOG_FILE"
python scripts/train.py "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training session completed successfully"
else
    echo "Training session failed with exit code: $EXIT_CODE"
fi
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log saved to: $LOG_FILE"
echo "================================================================================"

exit $EXIT_CODE
