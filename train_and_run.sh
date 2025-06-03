#!/bin/bash

VENV_DIR="venv"

SCRIPT_NAME="train_model.py"

if [ "$#" -lt 1 ]; then
    echo "❌ Usage: ./train_and_run.sh path/to/train.csv [path/to/test.csv] [num_samples] [split_ratio]"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"
NUM_SAMPLES="${3:-20}"      # default = 20
SPLIT_RATIO="${4:-0.3}"     # default = 0.3

if [ ! -d "$VENV_DIR" ]; then
    echo "🛠 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "⚙️ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

if [ -n "$TEST_FILE" ]; then
    echo "🚀 Running training with test set..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO"
else
    echo "🚀 Running training without test set (splitting train)..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO"
fi

deactivate
