#!/bin/bash

VENV_DIR="venv_fewshot"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creazione virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔧 Attivazione virtual environment..."
source "$VENV_DIR/bin/activate"

echo "📥 Installazione requirements..."
pip install --upgrade pip
pip install -r requirements_fewShot.txt

# Verifica se Ollama è installato
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama non trovato, installazione in corso..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama già installato."
fi

# Estrai modello se passato via --model, altrimenti default
MODEL_NAME="llama3.2:1b"
next_is_model=false

for arg in "$@"; do
    if [ "$next_is_model" = true ]; then
        MODEL_NAME="$arg"
        break
    fi
    if [ "$arg" = "--model" ]; then
        next_is_model=true
    fi
done

# Rimuovi i ":" per sicurezza nome file
MODEL_SAFE_NAME=$(echo "$MODEL_NAME" | tr ':' '_')

echo "🤖 Download del modello '$MODEL_NAME'..."
ollama pull "$MODEL_NAME"

# Cartella temporanea per output
OUTPUT_DIR="fewshot_outputs"
mkdir -p "$OUTPUT_DIR"

# Emozioni da ciclare
emotions=("positive" "negative" "neutral")

for emotion in "${emotions[@]}"; do
    echo "🚀 Esecuzione script Python per emotion: $emotion"
    python fewShot_generation.py "$@" --target_polarity "$emotion"
    
    # Sposta output generato nel folder temporaneo
    output_file="fewShot_generation_Ollama_${MODEL_SAFE_NAME}_${emotion}.json"
    if [ -f "$output_file" ]; then
        mv "$output_file" "$OUTPUT_DIR/"
    else
        echo "⚠️ Attenzione: File '$output_file' non trovato."
    fi
done

# Crea archivio ZIP
ZIP_FILE="fewShot_generations_${MODEL_SAFE_NAME}.zip"
zip -j "$ZIP_FILE" "$OUTPUT_DIR"/*.json

echo "✅ Archivio creato: $ZIP_FILE"
echo "📦 Contenuto:"
unzip -l "$ZIP_FILE"
