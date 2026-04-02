#!/bin/bash
# Train mertalizer v2 model with all improvements.
#
# Run this on a GPU machine (Lightning.ai Studio, Colab, etc.)
#
# Usage:
#   git clone <your-repo> && cd mertalizer
#   bash scripts/train_v2.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Mertalizer v2 Training ==="
echo ""

# # --- Setup ---
# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment..."
#     python3 -m venv venv
# fi
# source venv/bin/activate

echo "Installing dependencies..."
pip install -q torch torchaudio pytorch-lightning transformers \
    librosa soundfile numpy pandas scipy scikit-learn \
    omegaconf pyyaml tqdm 2>&1 | tail -3

# --- Verify GPU ---
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected. Training will be very slow.')
    print('On Lightning.ai: switch your Studio to GPU mode.')
"

# --- Storage path (override with TRAINING_STORAGE env var) ---
TRAINING_STORAGE="${TRAINING_STORAGE:-/teamspace/lightning_storage/mert-training-set}"
DATA_DIR="${TRAINING_STORAGE}"
echo "Data dir: ${DATA_DIR}"

# --- Verify data ---
if [ ! -f "${DATA_DIR}/splits/train.jsonl" ]; then
    echo "ERROR: Training data not found at ${DATA_DIR}/splits/"
    echo "Set TRAINING_STORAGE to your data root."
    exit 1
fi

TRAIN_TRACKS=$(wc -l < "${DATA_DIR}/splits/train.jsonl")
VAL_TRACKS=$(wc -l < "${DATA_DIR}/splits/validation.jsonl")
echo "Data: ${TRAIN_TRACKS} train / ${VAL_TRACKS} val tracks"

# --- Train ---
echo ""
echo "Starting training with v2 config..."
echo "  - TCN: kernel=15, layers=6, dilation=3 (68s receptive field)"
echo "  - Focal loss on labels + class weights"
echo "  - Data augmentation: crop, stretch, noise, dropout"
echo ""

export PYTHONPATH="$(pwd)/ml:${PYTHONPATH}"
python3 ml/training/train.py --config configs/mert_95m_v2.yaml --data-dir "${DATA_DIR}"

echo ""
echo "=== Training complete ==="

# --- Export ---
BEST_CKPT=$(ls -t models/checkpoints/best-*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="models/checkpoints/last.ckpt"
fi

echo "Exporting TorchScript head model from: $BEST_CKPT"
PYTHONPATH=ml python3 ml/export/torchscript.py \
    --checkpoint "$BEST_CKPT" \
    --output models/mertalizer_traced_v2.pt \
    --embed-dim 768

echo ""
echo "Done! New model at: models/mertalizer_traced_v2.pt"
echo "To use: export MODEL_PATH=models/mertalizer_traced_v2.pt"
