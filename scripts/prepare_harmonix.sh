#!/bin/bash
# Download Harmonix BigVGAN audio and extract MERT embeddings.
# Run this on Lightning.ai GPU Studio BEFORE training.
#
# Usage:
#   bash scripts/prepare_harmonix.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/ml:${PYTHONPATH}"

STORAGE="${TRAINING_STORAGE:-/teamspace/lightning_storage/mert-training-set}"

echo "=== Preparing Harmonix Dataset ==="

# --- Install deps ---
pip install -q huggingface_hub torch torchaudio transformers librosa soundfile numpy 2>&1 | tail -3

# --- Download annotations + BigVGAN audio ---
echo "Downloading Harmonix annotations and BigVGAN audio (~8.5GB)..."
python3 -c "
from huggingface_hub import hf_hub_download
for fname in ['harmonixset.corrected.20250821.jsonl', 'harmonixset_bigvgan.zip']:
    path = hf_hub_download(
        repo_id='m-a-p/harmonixset_bigvgan',
        filename=fname,
        repo_type='dataset',
        local_dir='data/raw/harmonix',
    )
    print(f'Downloaded: {path}')
"

echo "Extracting audio..."
cd data/raw/harmonix
unzip -qo harmonixset_bigvgan.zip
cd ../../..

AUDIO_DIR=$(find data/raw/harmonix -type d -name "harmonixset_bigvgan" | head -1)
if [ -z "$AUDIO_DIR" ]; then
    AUDIO_DIR="data/raw/harmonix"
fi
echo "Audio dir: $AUDIO_DIR"

# --- Re-ingest with audio paths ---
echo "Ingesting annotations with audio paths..."
python3 ml/data/ingest_harmonix.py \
    data/raw/harmonix/harmonixset.corrected.20250821.jsonl \
    --audio-dir "$AUDIO_DIR" \
    --output data/processed/harmonix.jsonl

# --- Extract MERT embeddings ---
echo "Extracting MERT embeddings (this takes ~1-2 hours on GPU)..."
python3 ml/data/preprocessing.py \
    --annotations data/processed/harmonix.jsonl \
    --output-dir data/processed/embeddings \
    --dataset-name harmonix \
    --model mert

# --- Merge splits ---
echo "Merging CCMusic + Harmonix splits..."
python3 -c "
import json
from pathlib import Path

storage = '${STORAGE}'
splits_dir = Path(storage) / 'splits'
splits_dir.mkdir(parents=True, exist_ok=True)

# Load existing CCMusic splits
for split_name in ['train', 'validation', 'test']:
    existing = []
    src = splits_dir / f'{split_name}.jsonl'
    if src.exists():
        with open(src) as f:
            existing = [json.loads(l) for l in f]
    # Filter out any old harmonix entries
    existing = [t for t in existing if t.get('dataset') != 'harmonix']

    # Load harmonix
    harmonix_split = {'train': 'train', 'validation': 'val', 'test': 'test'}[split_name]
    harmonix = []
    with open('data/processed/harmonix.jsonl') as f:
        for line in f:
            t = json.loads(line)
            if t['split'] == harmonix_split:
                harmonix.append(t)

    merged = existing + harmonix
    with open(src, 'w') as f:
        for t in merged:
            f.write(json.dumps(t) + '\n')
    print(f'{split_name}: {len(existing)} ccmusic + {len(harmonix)} harmonix = {len(merged)} total')

# Also copy embeddings to storage
import shutil
emb_src = Path('data/processed/embeddings/harmonix')
emb_dst = Path(storage) / 'embeddings' / 'harmonix'
if emb_src.exists():
    shutil.copytree(emb_src, emb_dst, dirs_exist_ok=True)
    print(f'Copied embeddings to {emb_dst}')
"

echo ""
echo "=== Done! Now run: bash scripts/train_v2.sh ==="
