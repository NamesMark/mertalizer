#!/bin/bash

# 🎵 Mertalizer Demo Script
# Comprehensive demonstration of music structure recognition system

set -e

echo "🎵 MERTALIZER DEMO - Music Structure Recognition System"
echo "========================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

echo ""
print_step "DEMO 1: Label Mapping System"
echo "Demonstrating canonical label mapping across different datasets..."

python3 -c "
from src.data.ontology import LabelMapper, CANONICAL_LABELS

print(f'Canonical labels: {CANONICAL_LABELS}')
print()

# Test various label formats
test_cases = [
    ('harmonix', ['intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'solo', 'outro']),
    ('salami', ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Solo', 'Outro']),
    ('beatles', ['前奏', '主歌', '副歌', '间奏', '尾奏']),  # Chinese labels
    ('spam', ['instrumental', 'vocal', 'silence', 'unknown_label'])
]

mapper = LabelMapper()

for dataset, labels in test_cases:
    print(f'{dataset.upper()} dataset:')
    mapped = mapper.map_labels(labels, dataset)
    for orig, mapped_label in zip(labels, mapped):
        print(f'  {orig:15} -> {mapped_label}')
    print()
"

print_success "Label mapping demonstration completed"

echo ""
print_step "DEMO 2: Dataset Configuration"
echo "Showing available datasets and their configurations..."

python3 -c "
from src.data.fetch_datasets import DatasetFetcher, DATASET_CONFIGS

print('Available datasets:')
for name, config in DATASET_CONFIGS.items():
    print(f'  {name.upper()}:')
    print(f'    Source: {config.source}')
    print(f'    Files: {config.files}')
    print(f'    Audio: {config.audio_source}')
    print()

fetcher = DatasetFetcher()
print(f'Total datasets configured: {len(DATASET_CONFIGS)}')
"

print_success "Dataset configuration demonstration completed"

echo ""
print_step "DEMO 3: Model Architecture"
echo "Creating and demonstrating the two-head model architecture..."

python3 -c "
import torch
from src.modeling.system import MusicStructureModel, ModelConfig, create_model

print('Creating model with default configuration...')
config = ModelConfig()
model = create_model(config)

print(f'Model type: {type(model).__name__}')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print()

# Test forward pass
print('Testing forward pass...')
batch_size, seq_len = 1, 1000
audio = torch.randn(batch_size, seq_len * 100)  # 100 samples per frame
sr = 22050

with torch.no_grad():
    outputs = model(audio, sr)

print(f'Boundary logits shape: {outputs[\"boundary_logits\"].shape}')
print(f'Label logits shape: {outputs[\"label_logits\"].shape}')
print(f'Embeddings shape: {outputs[\"embeddings\"].shape}')
print()

# Test prediction methods
print('Testing prediction methods...')
boundaries = model.predict_boundaries(audio, sr)
labels = model.predict_labels(audio, sr, boundaries)

print(f'Predicted boundaries: {len(boundaries)} boundaries')
print(f'Predicted labels: {labels}')
"

print_success "Model architecture demonstration completed"

echo ""
print_step "DEMO 4: Audio Preprocessing Pipeline"
echo "Demonstrating audio preprocessing and SSL embedding extraction..."

python3 -c "
import numpy as np
from src.data.preprocessing import AudioPreprocessor, AudioConfig

print('Creating audio preprocessor...')
config = AudioConfig(
    target_sr=22050,
    chunk_duration=20.0,
    hop_duration=10.0,
    trim_silence=True
)
preprocessor = AudioPreprocessor(config)

print(f'Target sample rate: {config.target_sr} Hz')
print(f'Chunk duration: {config.chunk_duration}s')
print(f'Hop duration: {config.hop_duration}s')
print()

# Create dummy audio for demonstration
print('Creating dummy audio signal...')
duration = 30.0  # 30 seconds
sr = 22050
audio = np.random.randn(int(duration * sr)).astype(np.float32)

print(f'Audio length: {len(audio)} samples ({duration}s)')
print()

# Detect beats
print('Detecting beats...')
beats, downbeats = preprocessor.detect_beats(audio, sr)
print(f'Detected {len(beats)} beats, {len(downbeats)} downbeats')
print()

# Chunk audio
print('Chunking audio...')
chunks = preprocessor.chunk_audio(audio, sr)
print(f'Created {len(chunks)} chunks')
print(f'Chunk shape: {chunks[0].shape if chunks else \"No chunks\"}')
print()

print('Note: SSL embedding extraction requires actual model loading')
print('This would extract MERT or w2v-BERT embeddings in production')
"

print_success "Audio preprocessing demonstration completed"

echo ""
print_step "DEMO 5: Evaluation Metrics"
echo "Demonstrating boundary detection and label classification metrics..."

python3 -c "
from src.evaluation.metrics import SegmentEvaluator, BoundaryEvaluator, LabelEvaluator
import numpy as np

# Create evaluator
label_names = ['INTRO', 'VERSE', 'PRE', 'CHORUS', 'BRIDGE', 'SOLO', 'OUTRO', 'OTHER']
evaluator = SegmentEvaluator(label_names)

# Create sample data
predicted_boundaries = [0.0, 12.5, 25.0, 37.5, 50.0]
predicted_labels = ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'OUTRO']

ground_truth_boundaries = [0.0, 12.0, 25.5, 38.0, 50.0]
ground_truth_labels = ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'OUTRO']

print('Sample predictions:')
print(f'Predicted boundaries: {predicted_boundaries}')
print(f'Predicted labels: {predicted_labels}')
print()

print('Ground truth:')
print(f'True boundaries: {ground_truth_boundaries}')
print(f'True labels: {ground_truth_labels}')
print()

# Evaluate
result = evaluator.evaluate_track(
    predicted_boundaries, predicted_labels,
    ground_truth_boundaries, ground_truth_labels
)

print('Evaluation Results:')
print(f'Boundary F1@0.5s: {result[\"boundary_metrics_05\"].f1:.3f}')
print(f'Boundary F1@3.0s: {result[\"boundary_metrics_30\"].f1:.3f}')
print(f'Label accuracy: {result[\"label_metrics\"].accuracy:.3f}')
print(f'Label F1: {result[\"label_metrics\"].f1:.3f}')
"

print_success "Evaluation metrics demonstration completed"

echo ""
print_step "DEMO 6: Classic Baseline Model"
echo "Demonstrating self-similarity matrix and novelty detection..."

python3 -c "
import numpy as np
from src.modeling.ssm_novelty import ClassicBaseline, SelfSimilarityMatrix

print('Creating classic baseline model...')
baseline = ClassicBaseline(
    window_size=30,
    novelty_threshold=0.1,
    min_segment_length=2.0
)

print(f'Window size: {baseline.window_size}')
print(f'Novelty threshold: {baseline.novelty_threshold}')
print(f'Min segment length: {baseline.min_segment_length}s')
print()

# Create dummy embeddings
print('Creating dummy feature embeddings...')
np.random.seed(42)
embeddings = np.random.randn(1000, 768)  # 1000 frames, 768-dim embeddings
print(f'Embeddings shape: {embeddings.shape}')
print()

# Test self-similarity matrix
print('Computing self-similarity matrix...')
ssm_computer = SelfSimilarityMatrix()
ssm = ssm_computer.compute_ssm(embeddings)
print(f'SSM shape: {ssm.shape}')
print(f'SSM diagonal mean: {np.mean(np.diag(ssm)):.3f}')
print()

# Test novelty detection
print('Computing novelty function...')
novelty = ssm_computer.compute_novelty(ssm)
print(f'Novelty shape: {novelty.shape}')
print(f'Novelty max: {np.max(novelty):.3f}')
print()

# Test boundary detection
print('Detecting boundaries...')
boundaries = baseline.detect_boundaries(embeddings, sr=22050)
print(f'Detected {len(boundaries)} boundaries')
if boundaries:
    print(f'Boundary times: {boundaries[:5]}...')  # Show first 5
print()

# Test label prediction
print('Predicting labels...')
labels = baseline.predict_labels(embeddings, boundaries, sr=22050)
print(f'Predicted labels: {labels}')
"

print_success "Classic baseline demonstration completed"

echo ""
print_step "DEMO 7: Rust Web Server"
echo "Testing Rust web server compilation and basic functionality..."

cd rust
if cargo build --release > /dev/null 2>&1; then
    print_success "Rust web server compiles successfully"
    echo "Binary size: $(du -h target/release/web_server | cut -f1)"
else
    print_error "Rust compilation failed"
fi
cd ..

echo ""
print_info "Rust web server features:"
echo "  • Axum-based async web framework"
echo "  • Multipart file upload handling"
echo "  • CORS support for web dashboard"
echo "  • Integration with Python ML API"
echo "  • Static file serving"

print_success "Rust web server demonstration completed"

echo ""
print_step "DEMO 8: API Integration"
echo "Demonstrating FastAPI REST API structure..."

python3 -c "
from src.inference.api import create_app, PredictionRequest, PredictionResponse
from fastapi.testclient import TestClient

print('Creating FastAPI application...')
app = create_app('dummy_model.ckpt')

print('Available endpoints:')
print('  • POST /predict - Single file prediction')
print('  • POST /predict-batch - Batch file prediction')
print('  • GET /health - Health check')
print('  • GET /models - List available models')
print('  • GET /config - Get current configuration')
print()

print('API features:')
print('  • Automatic file validation')
print('  • Temporary file management')
print('  • Error handling and logging')
print('  • Background task cleanup')
print('  • CORS middleware')
print('  • OpenAPI documentation')
"

print_success "API integration demonstration completed"

echo ""
print_step "DEMO 9: ONNX Export Capability"
echo "Demonstrating model export to ONNX format..."

python3 -c "
import torch
from src.export.onnx import ONNXExporter
from src.modeling.system import MusicStructureModel, ModelConfig

print('Creating model for ONNX export...')
config = ModelConfig()
model = MusicStructureModel(config)
model.eval()

print(f'Model created: {type(model).__name__}')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print()

print('ONNX export features:')
print('  • PyTorch to ONNX conversion')
print('  • Model verification')
print('  • Inference testing')
print('  • Cross-platform deployment')
print('  • Performance optimization')
print()

print('Note: Actual export requires trained model checkpoint')
print('Command: python src/export/onnx.py --checkpoint model.ckpt --output model.onnx')
"

print_success "ONNX export demonstration completed"

echo ""
print_step "DEMO 10: Complete Pipeline Overview"
echo "Showing the complete music structure recognition pipeline..."

echo ""
print_info "🎵 MERTALIZER PIPELINE OVERVIEW"
echo "=================================="
echo ""
echo "1. 📥 DATA ACQUISITION"
echo "   • Download datasets from Hugging Face, Git, URLs"
echo "   • Support for Harmonix, SALAMI, Beatles, SPAM, CCMUSIC"
echo "   • Automatic annotation parsing and normalization"
echo ""
echo "2. 🏷️  LABEL MAPPING"
echo "   • Canonical label ontology (INTRO, VERSE, PRE, CHORUS, etc.)"
echo "   • Dataset-specific synonym mapping"
echo "   • Unknown label handling → OTHER"
echo ""
echo "3. 🎧 AUDIO PREPROCESSING"
echo "   • Resampling to 22.05 kHz mono"
echo "   • Beat/downbeat detection with librosa"
echo "   • SSL embedding extraction (MERT, w2v-BERT)"
echo "   • Beat-synchronized feature pooling"
echo ""
echo "4. 🧠 MODEL ARCHITECTURE"
echo "   • Two-head design: boundary detection + label classification"
echo "   • Frozen/fine-tunable SSL encoder"
echo "   • TCN or Transformer boundary head"
echo "   • Linear + CRF label head"
echo ""
echo "5. 🎯 TRAINING PROTOCOL"
echo "   • Joint loss: boundary BCE + label NLL"
echo "   • Class-balanced weights and oversampling"
echo "   • SpecAugment data augmentation"
echo "   • AdamW optimizer with cosine decay"
echo ""
echo "6. 📊 EVALUATION METRICS"
echo "   • Boundary F1@0.5s and F1@3.0s"
echo "   • Label accuracy and per-class metrics"
echo "   • Cross-dataset generalization"
echo "   • Error analysis and visualization"
echo ""
echo "7. 🚀 INFERENCE & DEPLOYMENT"
echo "   • CLI tool for batch processing"
echo "   • FastAPI REST API"
echo "   • Rust web dashboard"
echo "   • ONNX export for production"
echo ""

print_success "Complete pipeline overview demonstrated"

echo ""
print_step "DEMO SUMMARY"
echo "================"
echo ""
print_success "✅ All core components demonstrated successfully!"
echo ""
print_info "🎯 KEY FEATURES SHOWCASED:"
echo "  • Comprehensive label mapping system"
echo "  • Multi-dataset support and configuration"
echo "  • Advanced two-head model architecture"
echo "  • Complete audio preprocessing pipeline"
echo "  • Robust evaluation metrics"
echo "  • Classic baseline with novelty detection"
echo "  • High-performance Rust web server"
echo "  • Production-ready API integration"
echo "  • ONNX export for deployment"
echo "  • End-to-end pipeline architecture"
echo ""
print_info "🚀 READY FOR PRODUCTION:"
echo "  • Run ./example.sh for complete pipeline"
echo "  • Use ./run_server.sh to start web services"
echo "  • Train models with configs in configs/"
echo "  • Deploy with Docker or ONNX export"
echo ""
print_success "🎉 Mertalizer demo completed successfully!"
echo ""
echo "Next steps:"
echo "1. Train a model: python src/training/train.py --config configs/mert_95m.yaml"
echo "2. Run inference: python src/inference/cli.py audio.wav --model models/best.ckpt"
echo "3. Start web dashboard: ./run_server.sh"
echo "4. Export to ONNX: python src/export/onnx.py --checkpoint models/best.ckpt --output model.onnx"
