# RoBERTa 

Questo progetto esegue **fine-tuning di un modello RoBERTa** per la classificazione del sentiment (es. positivo, neutro, negativo) su dati testuali utilizzando **python 3.11**.
Il training e la predizione vengono gestiti tramite uno script Python (`train_and_predict.py`) e uno script Bash (`run_train.sh`).

---

## 📁 Contenuto del repository

- `train_and_predict.py`: script principale per addestramento e predizione.
- `run_train.sh`: script per automatizzare creazione ambiente virtuale e esecuzione.
- `requirements.txt`: dipendenze necessarie per eseguire il progetto.
- `test_predictions.csv`: file CSV generato con le predizioni sul test set.

---

## 🚀 Avvio rapido

### 1. 📋 Prepara i file

Prepara due file CSV con separatore `;`:

- **Train file**: deve contenere almeno le colonne `Text` e `Polarity`.
- **Test file**: deve contenere `Text` (e opzionalmente `ID`, `Polarity`).

### 2. ▶️ Esecuzione

Rendi eseguibile lo script bash e lancialo passando i percorsi ai CSV:

```bash
chmod +x run_train.sh
./run_train.sh path/to/train.csv path/to/test.csv
