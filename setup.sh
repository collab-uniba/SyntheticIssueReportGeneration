#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_tesi"

echo "Creazione virtual environment..."
python3 -m venv "$VENV_DIR"

echo "Attivazione virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Aggiornamento pip..."
pip install --upgrade pip setuptools wheel

echo "Installazione PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installazione requirements..."
pip install -r requirements_locked.txt

echo "Controllo disponibilità GPU e coerenza environment..."
python - <<'PY'
import torch, transformers, datasets, setfit, pandas, numpy, sklearn
print("IMPORT OK")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Setup completato!"
echo "Per attivare l'ambiente: source venv_tesi/bin/activate"