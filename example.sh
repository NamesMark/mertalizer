#!/bin/bash

# Mertalizer Example Script
# Demonstrates the complete pipeline from data fetching to web dashboard

set -e

echo "🎵 Mertalizer Complete Example Pipeline"
echo "======================================="

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Step 1: Setup
echo ""
echo "📦 Step 1: Setup"
echo "---------------"
if [ ! -d "venv" ]; then
    echo "Running setup script..."
    ./setup.sh
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Step 2: Download datasets (if not already done)
echo ""
echo "📥 Step 2: Download Datasets"
echo "----------------------------"
if [ ! -d "data/raw/harmonix" ]; then
    echo "Downloading Harmonix dataset..."
    python src/data/fetch_datasets.py --dataset harmonix
else
    echo "✅ Harmonix dataset already downloaded"
fi

# Step 3: Process data (if not already done)
echo ""
echo "🔄 Step 3: Process Data"
echo "-----------------------"
if [ ! -f "data/processed/combined_annotations.csv" ]; then
    echo "Processing datasets..."
    python src/data/ingestion.py --all
else
    echo "✅ Data already processed"
fi

# Step 4: Extract embeddings (if not already done)
echo ""
echo "🧠 Step 4: Extract SSL Embeddings"
echo "---------------------------------"
if [ ! -d "data/processed/embeddings" ]; then
    echo "Extracting embeddings..."
    # This would normally process all audio files
    # For demo purposes, we'll create a dummy model
    echo "Creating dummy model for demonstration..."
    mkdir -p models/checkpoints
    python -c "
import torch
from src.modeling.system import create_model, ModelConfig

config = ModelConfig()
model = create_model(config)
torch.save(model.state_dict(), 'models/checkpoints/dummy_model.ckpt')
print('Dummy model created')
"
else
    echo "✅ Embeddings already extracted"
fi

# Step 5: Train model (if not already done)
echo ""
echo "🎯 Step 5: Train Model"
echo "----------------------"
if [ ! -f "models/best_model.ckpt" ]; then
    echo "Training model..."
    # For demo, we'll create a dummy checkpoint
    python -c "
import torch
from src.modeling.system import create_model, ModelConfig

config = ModelConfig()
model = create_model(config)
torch.save(model.state_dict(), 'models/best_model.ckpt')
print('Dummy trained model created')
"
else
    echo "✅ Model already trained"
fi

# Step 6: Export to ONNX (optional)
echo ""
echo "📤 Step 6: Export to ONNX"
echo "-------------------------"
if [ ! -f "models/model.onnx" ]; then
    echo "Exporting model to ONNX..."
    python src/export/onnx.py --checkpoint models/best_model.ckpt --output models/model.onnx
else
    echo "✅ ONNX model already exported"
fi

# Step 7: Test CLI inference
echo ""
echo "🔧 Step 7: Test CLI Inference"
echo "-----------------------------"
echo "Testing CLI with dummy audio..."
# Create a dummy audio file for testing
python -c "
import numpy as np
import soundfile as sf

# Create 10 seconds of dummy audio
sr = 22050
duration = 10
t = np.linspace(0, duration, sr * duration)
audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz sine wave
sf.write('test_audio.wav', audio, sr)
print('Test audio file created: test_audio.wav')
"

# Test CLI
python src/inference/cli.py test_audio.wav --model models/best_model.ckpt --output test_results.json
echo "✅ CLI inference test completed"

# Step 8: Start servers
echo ""
echo "🚀 Step 8: Start Web Servers"
echo "----------------------------"
echo "Starting Python API and Rust web server..."
echo "This will run in the background. Press Ctrl+C to stop."

# Start servers in background
./run_server.sh &
SERVER_PID=$!

# Wait a bit for servers to start
sleep 10

# Test API
echo "Testing API endpoint..."
curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"

echo ""
echo "🎉 Example pipeline completed!"
echo ""
echo "🌐 Web Dashboard: http://localhost:3000"
echo "📊 Python API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "You can now:"
echo "1. Upload audio files through the web dashboard"
echo "2. Use the CLI for batch processing"
echo "3. Integrate with the REST API"
echo ""
echo "Press Ctrl+C to stop the servers"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $SERVER_PID 2>/dev/null || true
    rm -f test_audio.wav test_results.json
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
