#!/bin/bash

VENV_DIR="venv"
SCRIPT_NAME="train_model.py"

TEST_FILE="test_stackoverflowOriginali.csv"
PRED_DIR="predictions"
ZIP_NAME="all_predictions.zip"

TRAIN_FILES=(
    "Dataset_fewShot_combined.csv"
    "Dataset_zeroShot_combined.csv"
    "fewShot_generation.csv"
    "train_stackoverflowOriginale.csv"
    "zeroShot_generation.csv"
)

# Pulizia output precedente
rm -rf "$PRED_DIR" "$ZIP_NAME"
mkdir -p "$PRED_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "🛠 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "⚙️ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

for TRAIN_FILE in "${TRAIN_FILES[@]}"; do
    echo "🚀 Running training with:"
    echo "    ▶️ Train: $TRAIN_FILE"
    echo "    🧪 Test : $TEST_FILE"

    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE"

    BASE_NAME=$(basename "$TRAIN_FILE" .csv)
    OUTPUT_FILE="test_predictions_${BASE_NAME}.csv"

    if [ -f "$OUTPUT_FILE" ]; then
        mv "$OUTPUT_FILE" "$PRED_DIR/"
    else
        echo "⚠️ Warning: Expected output $OUTPUT_FILE not found!"
    fi
done

echo "📦 Zipping prediction files into $ZIP_NAME..."
zip -j "$ZIP_NAME" "$PRED_DIR"/*.csv

deactivate

echo "✅ Done! All predictions zipped in: $ZIP_NAME"
