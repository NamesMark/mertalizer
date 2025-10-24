#!/bin/bash

# Mertalizer Server Startup Script
# Starts both Python API and Rust web server

set -e

echo "🚀 Starting Mertalizer servers..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "📦 Activating Python virtual environment..."
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Ensure PYTHONPATH includes project src
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH"
    echo "Please train a model first or update MODEL_PATH in .env"
    echo "To train: python src/training/train.py --config configs/mert_95m.yaml"
    exit 1
fi

# Start Python API server in background
echo "🐍 Starting Python API server on port 8000..."
python src/inference/api.py --model "$MODEL_PATH" --port 8000 &
PYTHON_PID=$!

# Verify API started
sleep 2
if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
    echo "❌ Failed to start Python API. Check logs above."
    exit 1
fi

# Wait for Python server to start
sleep 5

# Start Rust web server
echo "🦀 Starting Rust web server on port ${PORT:-3000}..."
cd rust
cargo run --bin web_server &
RUST_PID=$!
cd ..

echo ""
echo "🎉 Servers started!"
echo "📊 Python API: http://localhost:8000"
echo "🌐 Web Dashboard: http://localhost:${PORT:-3000}"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $PYTHON_PID 2>/dev/null || true
    kill $RUST_PID 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
