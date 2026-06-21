# ModernBERT

Questo progetto esegue fine-tuning di un modello ModernBERT per la classificazione del sentiment (es. positivo, neutro, negativo) su dati testuali.
Il training e la predizione vengono gestiti tramite uno script Python (train_modernBERT.py) e uno script Bash (run_train_modernbert.sh).

## Contenuto del repository
- train_modernbert.py: script principale per addestramento e predizione.
- run_train_modernbert.sh: script per automatizzare creazione ambiente virtuale e esecuzione.
- ModernBERT_results/: cartella che contiene tutte le predizioni e le metriche generate dagli esperimenti.

## Preparazione
Assicurati che i tuoi file train.csv e test.csv abbiano almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_train_modernbert.sh
- ./run_train_modernbert.sh train_*.csv test_*.csv (opzionale)

## Outputs
- Il file .csv e .json vengono salvati in: ModernBERT_outputs/
