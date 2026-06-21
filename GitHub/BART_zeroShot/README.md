# BART ZeroShot

Questo progetto esegue le predictions di un modello BART ZeroShot senza pre-training per la classificazione del sentiment (es. positivo, neutro, negativo) sul dataset di testset.
Il training e la predizione vengono gestiti tramite uno script Python (pred_bartzeroshot.py) e uno script Bash (run_bartzeroshot.sh).

## Contenuto del repository
- pred_bartzeroshot.py: script principale per predizioni.
- run_bartzeroshot.sh: script per automatizzare creazione ambiente virtuale e esecuzione.
- BART_zeroShot_results/: cartella che contiene predizioni e le metriche generate dall'esperimento.

## Preparazione
Assicurati che il  file test.csv abbia almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_bartzeroshot.sh
- ./run_bartzeroshot.sh test_*.csv

## Outputs
- Il file .csv e .json vengono salvati in: BART_zeroShot_outputs/
