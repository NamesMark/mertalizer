# Mertalizer: Music Structure Recognition

A comprehensive system for detecting musical structure boundaries and assigning section labels (intro/verse/pre-chorus/chorus/bridge/solo/outro/other) using music-native SSL encoders.

## 🎵 Features

- **Primary Model**: MERT 95M/330M SSL encoder with TCN/Transformer boundary detection and CRF labeling
- **Baseline Model**: w2v-BERT-2.0 for comparison
- **Datasets**: Harmonix Set, SALAMI, Beatles/Isophonics, SPAM, CCMUSIC
- **Outputs**: Trained checkpoints, CLI & REST inference, evaluation reports
- **Web Dashboard**: Interactive audio upload and visualization interface
- **Rust Integration**: High-performance web server and audio processing

## 🏗️ Project Structure

```
mertalizer/
├── src/                    # Python ML components
│   ├── data/              # Data processing and ingestion
│   │   ├── ontology.py    # Label mapping system
│   │   ├── fetch_datasets.py  # Dataset downloader
│   │   ├── ingestion.py   # Data normalization
│   │   └── preprocessing.py   # Audio preprocessing
│   ├── modeling/          # Model architectures
│   │   ├── system.py      # Two-head architecture
│   │   └── ssm_novelty.py # Classic baseline
│   ├── training/          # Training scripts
│   │   └── train.py       # LightningModule training
│   ├── evaluation/        # Metrics and evaluation
│   │   └── metrics.py     # Boundary & label metrics
│   ├── inference/         # CLI and REST API
│   │   ├── cli.py         # Command-line interface
│   │   └── api.py         # FastAPI REST server
│   └── export/            # ONNX export
│       └── onnx.py        # Model export utilities
├── rust/                  # Rust components
│   ├── web/               # Web server
│   │   └── main.rs       # Axum web server
│   └── Cargo.toml         # Rust dependencies
├── configs/               # Model configurations
│   ├── mert_95m.yaml     # MERT configuration
│   └── w2v_baseline.yaml # w2v-BERT configuration
├── web_dashboard/         # Frontend dashboard
│   └── templates/         # HTML templates
├── data/                  # Dataset storage
├── models/                # Trained checkpoints
└── scripts/              # Setup and run scripts
```

## 🚀 Quick Start

### Option 1: Complete Example Pipeline
```bash
# See all features in action
./demo.sh

# Run the complete example pipeline
./example.sh
```

### Option 2: Manual Setup
```bash
# 1. Setup environment
./setup.sh

# 2. Download datasets
python src/data/fetch_datasets.py --all

# 3. Process data
python src/data/ingestion.py --all

# 4. Train model
python src/training/train.py --config configs/mert_95m.yaml

# 5. Start web servers
./run_server.sh
```

## 🔧 Environment Variables

Create a `.env` file with:
```bash
# Hugging Face token for dataset access
HF_TOKEN=your_hf_token_here

# Weights & Biases API key (optional)
WANDB_KEY=your_wandb_key_here

# Model paths
MODEL_PATH=models/best_model.ckpt
PYTHON_API_URL=http://localhost:8000

# Server settings
PORT=3000
```

## 📊 Usage

### Web Dashboard
- **URL**: http://localhost:3000
- Upload audio files and visualize structure analysis
- Interactive waveform with segment boundaries
- Real-time structure charts and statistics

### CLI Inference
```bash
# Single file
python src/inference/cli.py audio.wav --model models/best_model.ckpt

# Batch processing
python src/inference/cli.py audio_dir/ --model models/best_model.ckpt --batch

# Export results
python src/inference/cli.py audio.wav --model models/best_model.ckpt --output results.json
```

### REST API
```bash
# Health check
curl http://localhost:8000/health

# Predict structure
curl -X POST -F "audio_file=@audio.wav" http://localhost:8000/predict

# API documentation
open http://localhost:8000/docs
```

### ONNX Export
```bash
# Export trained model to ONNX
python src/export/onnx.py --checkpoint models/best_model.ckpt --output models/model.onnx

# Test ONNX model
python src/export/onnx.py --checkpoint models/best_model.ckpt --output models/model.onnx --test
```

## 🎯 Model Architecture

### Two-Head Architecture
- **Encoder**: Frozen/fine-tunable SSL encoder (MERT or w2v-BERT)
- **Boundary Head**: TCN or Transformer for boundary detection
- **Label Head**: Linear classifier with CRF for segment labeling
- **Loss**: Combined boundary (BCE + Focal) and label (CrossEntropy) losses

### Classic Baseline
- **Self-Similarity Matrix**: Cosine similarity between embeddings
- **Novelty Detection**: Checkerboard kernel convolution
- **Peak Detection**: Threshold-based boundary detection
- **Label Assignment**: Majority vote or heuristic rules

## 📈 Evaluation Metrics

- **Boundary Detection**: F@0.5s, F@3.0s, Hit Rate
- **Label Classification**: Accuracy, Precision, Recall, F1
- **Cross-Dataset**: Generalization across different datasets
- **Error Analysis**: Confusion matrices and case studies

## 🔬 Datasets

| Dataset | Source | Annotations | Audio |
|---------|--------|-------------|-------|
| Harmonix Set | Hugging Face | Structure + Beats | Provided |
| SALAMI | GitHub | Hierarchical | External IDs |
| Beatles/Isophonics | Hugging Face | Structure + Chords | External |
| SPAM | GitHub | Multi-annotator | External |
| CCMUSIC | Hugging Face | Chinese Pop | External |

## 🛠️ Development

### Adding New Datasets
1. Update `src/data/fetch_datasets.py` with dataset configuration
2. Implement parser in `src/data/ingestion.py`
3. Add label mappings in `src/data/ontology.py`

### Custom Models
1. Create new model class in `src/modeling/`
2. Update `src/training/train.py` to support new architecture
3. Add configuration in `configs/`

### Web Dashboard Features
1. Modify `web_dashboard/templates/index.html`
2. Update Rust web server in `rust/web/main.rs`
3. Add new API endpoints in `src/inference/api.py`

## 📝 Output Format

```json
{
  "track_id": "example_track",
  "sr": 22050,
  "duration": 180.5,
  "boundaries": [0.0, 12.5, 25.0, 37.5, 50.0],
  "labels": ["INTRO", "VERSE", "CHORUS", "VERSE", "OUTRO"],
  "segments": [
    {"start": 0.0, "end": 12.5, "label": "INTRO"},
    {"start": 12.5, "end": 25.0, "label": "VERSE"},
    {"start": 25.0, "end": 37.5, "label": "CHORUS"},
    {"start": 37.5, "end": 50.0, "label": "VERSE"},
    {"start": 50.0, "end": 180.5, "label": "OUTRO"}
  ],
  "version": "mert-95m-tcn-crf@2025-01-24"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - Educational/Research Use

## 🙏 Acknowledgments

- MERT team for the SSL encoder
- Hugging Face for dataset hosting
- PyTorch Lightning for training framework
- Axum for Rust web framework