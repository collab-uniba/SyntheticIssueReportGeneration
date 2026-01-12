# RoBerta 

Questo progetto esegue fine-tuning di un modello RoBERTa per la classificazione del sentiment (es. positivo, neutro, negativo) su dati testuali utilizzando python 3.12.
Il training e la predizione vengono gestiti tramite uno script Python (train_and_predict.py) e uno script Bash (run_train.sh).

## Contenuto del repository
- train_and_predict.py: script principale per addestramento e predizione.
- run_train.sh: script per automatizzare creazione ambiente virtuale e esecuzione.
- requirements.txt: dipendenze necessarie per eseguire il progetto.
- RoBerta_outputs/: cartella che contiene il file test_predictions.csv generato con le predizioni sul test set.
- RoBerta_outputs.zip: archivio ZIP finale con i risultati.

## Preparazione
Prepara due file CSV con separatore ';':
- Train file: deve contenere almeno le colonne 'Text' e 'Polarity'.
- Test file: deve contenere 'Text' (e opzionalmente 'ID', 'Polarity').

## Esecuzione
'''bash

- chmod +x run_train.sh
- ./run_train.sh path/to/train.csv path/to/test.csv

## Outputs
- Il file .csv viene salvato in: RoBerta_outputs/
- Al termine viene creato: RoBerta_outputs.zip
