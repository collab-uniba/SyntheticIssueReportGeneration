#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="train_model.py"

TEST_FILE="test_stackoverflowOriginali.csv"
OUTPUT_DIR="Predictions"

TRAIN_FILES=(
    "fewShot_combined.csv"
    "zeroShot_combined.csv"
    "fewShot_output.csv"
    "train_stackoverflowOriginale.csv"
    "zeroShot_output.csv"
)

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# crea cartella output
mkdir -p "$OUTPUT_DIR"

# esecuzione script Python
for TRAIN_FILE in "${TRAIN_FILES[@]}"; do

    echo ""
    echo "Running training:"
    echo "    Train: $TRAIN_FILE"
    echo "    Test : $TEST_FILE"

    python "$SCRIPT_NAME" \
        -d "$TRAIN_FILE" \
        -t "$TEST_FILE"

    BASE_NAME=$(basename "$TRAIN_FILE" .csv)
    OUTPUT_FILE="test_predictions_${BASE_NAME}.csv"

    if [ -f "$OUTPUT_FILE" ]; then
        mv "$OUTPUT_FILE" "$OUTPUT_DIR/"
        echo "Salvato: $OUTPUT_DIR/$OUTPUT_FILE"
    else
        echo "Output non trovato: $OUTPUT_FILE"
    fi

done

echo ""
echo "Esecuzione completata. Risultati salvati in: $OUTPUT_DIR"