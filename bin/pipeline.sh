#!/bin/bash
# train -> evaluate -> predict -> explain を一連で実行するパイプライン
# Usage: ./bin/pipeline.sh [output_log_file]

set -e

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="outputs/logs"
LOG_FILE="${1:-${LOG_DIR}/pipeline_${TIMESTAMP}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "================================================================================"
echo "Starting pipeline (train -> evaluate -> predict -> explain)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file: $LOG_FILE"
echo "================================================================================"

# last.pt / best.pt を削除してフレッシュスタート
if [ -f "outputs/checkpoints/last.pt" ]; then
    echo "Removing existing last.pt to start fresh training..."
    rm outputs/checkpoints/last.pt
fi
if [ -f "outputs/checkpoints/best.pt" ]; then
    echo "Removing existing best.pt to start fresh training..."
    rm outputs/checkpoints/best.pt
fi

{
    # --- train ---
    echo "=== [1/4] train ==="
    TRAIN_OUT=$(python scripts/train.py 2>&1 | tee /dev/stderr)
    RUN_ID=$(echo "$TRAIN_OUT" | grep -E '^[0-9a-f]{32}$' | head -1)
    echo "run_id: $RUN_ID"

    CKPT_DIR=$(find mlruns -path "*/${RUN_ID}/artifacts/checkpoints" | head -1)
    echo "ckpt_dir: $CKPT_DIR"

    # --- evaluate ---
    echo "=== [2/4] evaluate ==="
    python scripts/evaluate.py evaluate.run_id="$RUN_ID" train.ckpt.dir="$CKPT_DIR"

    # --- predict ---
    # case_dirが存在しない場合はpredict.py側でd_inに合うケースを自動選択する
    echo "=== [3/4] predict ==="
    python scripts/predict.py predict.run_id="$RUN_ID" train.ckpt.dir="$CKPT_DIR" train.ckpt.last=best.pt

    # --- explain ---
    echo "=== [4/4] explain ==="
    python scripts/explain.py explain.run_id="$RUN_ID" train.ckpt.dir="$CKPT_DIR"

    echo "Pipeline completed. run_id=$RUN_ID"
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully"
else
    echo "Pipeline failed with exit code: $EXIT_CODE"
fi
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log saved to: $LOG_FILE"
echo "================================================================================"

exit $EXIT_CODE
