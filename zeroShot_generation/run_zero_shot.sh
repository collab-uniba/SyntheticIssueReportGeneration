#!/bin/bash

VENV_DIR="venv_zeroshot"

if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creo virtual environment in '$VENV_DIR'..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "📚 Installo i pacchetti da requirements_zeroShot.txt..."
pip install --upgrade pip > /dev/null
pip install -r requirements_zeroShot.txt

# Verifica se Ollama è installato
if ! command -v ollama &> /dev/null; then
  echo "📦 Ollama non trovato, installazione in corso..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "✅ Ollama già installato."
fi

# Estrai nome modello (default: llama3.2:1b)
MODEL_NAME="llama3.2:1b"
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--model" ]]; then
    next_index=$((i + 1))
    MODEL_NAME="${!next_index}"
    break
  fi
done

# Rimuovi i ":" per nome file valido
MODEL_SAFE_NAME=$(echo "$MODEL_NAME" | tr ':' '_')

echo "🤖 Scarico il modello: $MODEL_NAME..."
ollama pull "$MODEL_NAME"

# Directory temporanea per output
OUTPUT_FILES=()

# Esegui per ogni emozione
for EMOTION in positive negative neutral; do
  echo "🚀 Avvio generazione zero-shot con emozione: $EMOTION..."
  python zeroShot_generation.py "$@" --emotion "$EMOTION"

  OUTPUT_FILE="zero_shot_generation_${MODEL_SAFE_NAME}_${EMOTION}.json"
  if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_FILES+=("$OUTPUT_FILE")
  else
    echo "⚠️  Attenzione: file $OUTPUT_FILE non trovato, potrebbe esserci stato un errore."
  fi
done

# Crea archivio zip
ZIP_NAME="zero_shot_outputs_${MODEL_SAFE_NAME}.zip"
if [ ${#OUTPUT_FILES[@]} -gt 0 ]; then
  echo "📦 Creo archivio ZIP: $ZIP_NAME con i file JSON generati..."
  zip -j "$ZIP_NAME" "${OUTPUT_FILES[@]}"
  echo "✅ Archivio creato: $ZIP_NAME"
else
  echo "❌ Nessun file JSON da zippare."
fi
