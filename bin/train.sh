#!/bin/bash
# 新規学習を開始するスクリプト
# Usage: ./bin/train.sh [output_log_file]

set -e

# ログファイル名を指定（デフォルトは自動生成）
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="outputs/logs"
LOG_FILE="${1:-${LOG_DIR}/train_${TIMESTAMP}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "================================================================================"
echo "Starting new training session"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: $LOG_FILE"
echo "================================================================================"

# last.pt
if [ -f "outputs/checkpoints/last.pt" ]; then
    echo "Removing existing last.pt to start fresh training..."
    rm outputs/checkpoints/last.pt
fi

# best.pt
if [ -f "outputs/checkpoints/best.pt" ]; then
    echo "Removing existing best.pt to start fresh training..."
    rm outputs/checkpoints/best.pt
fi

# トレーニング実行（標準出力とエラー出力を両方ファイルに保存）
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

