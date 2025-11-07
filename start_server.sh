#!/bin/bash
# Start Mertalizer server with native Rust inference

set -e

cd "$(dirname "$0")"

echo "🚀 Starting Mertalizer with native Rust inference..."

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Run ./scripts/setup.sh first"
    exit 1
fi

# Set up libtorch
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib

# Check model exists
if [ ! -f "models/mertalizer_traced.pt" ]; then
    echo "⚠️  TorchScript model not found. Exporting..."
    PYTHONPATH=ml python ml/export/torchscript.py \
        --checkpoint models/checkpoints/mertalizer_final_model.ckpt \
        --output models/mertalizer_traced.pt \
        --embed-dim 768
fi

# Optional: Set log level
export RUST_LOG=${RUST_LOG:-info}

# Run server
echo "✅ Starting server on http://0.0.0.0:3000"
echo "   Press Ctrl+C to stop"
echo ""

cargo run --release

