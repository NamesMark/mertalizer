#!/bin/bash

# Mertalizer Setup Script
# Sets up the music structure recognition system

set -e

echo "🎵 Setting up Mertalizer - Music Structure Recognition System"
echo "=============================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is required but not installed."
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "✅ Python and Rust are available"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Build Rust components
echo "🦀 Building Rust web server..."
cd rust
cargo build --release
cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,splits,embeddings}
mkdir -p models/checkpoints
mkdir -p uploads
mkdir -p logs

# Set up environment variables
echo "🔧 Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << EOF
# Hugging Face token for dataset access
HF_TOKEN=your_hf_token_here

# Weights & Biases API key (optional)
WANDB_KEY=your_wandb_key_here

# Model paths
MODEL_PATH=models/best_model.ckpt
PYTHON_API_URL=http://localhost:8000

# Server settings
PORT=3000
EOF
    echo "📝 Created .env file. Please update with your API keys."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Download datasets: python src/data/fetch_datasets.py --all"
echo "3. Process data: python src/data/ingestion.py --all"
echo "4. Train model: python src/training/train.py --config configs/mert_95m.yaml"
echo "5. Start web server: ./run_server.sh"
echo ""
echo "For more information, see README.md"
