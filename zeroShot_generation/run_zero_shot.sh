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

MODEL_NAME="llama3.2:1b"

for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--model" ]]; then
    next_index=$((i + 1))
    MODEL_NAME="${!next_index}"
    break
  fi
done

echo "🤖 Scarico il modello: $MODEL_NAME..."
ollama pull "$MODEL_NAME"

echo "🚀 Avvio generazione zero-shot..."
python zeroShot_generation.py "$@"
