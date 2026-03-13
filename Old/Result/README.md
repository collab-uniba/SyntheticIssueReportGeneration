# Addestramento finale e analisi dei risultati

Questo progetto utilizza SetFit (Sentence Transformer Fine-tuning) per addestrare modelli di classificazione del sentiment e generare un resoconto finale dei risultati.

Lo script train_model.py viene eseguito tramite run_all_trainings.sh, permettendo di confrontare diversi dataset (originale, few-shot, zero-shot, combinati) e calcolare metriche standard di classificazione, generando anche file CSV con le predizioni e report JSON.

## Contenuto del Repository
- train_model.py: Script principale per addestrare SetFit, effettuare predizioni e generare il report finale.
- run_all_trainings.sh: Script per gestire l'ambiente virtuale, installare dipendenze e lanciare python su più dataset automaticamente.
- requirements.txt: Lista delle dipendenze necessarie.
- fewShot_output.csv: dataset generato tramite modello fewShot.
- zeroShot_output.csv: datset generato tramite modello zeroShot.
- fewShot_combined.csv: dataset ottenuto dalla concatenazione di fewShot con dataset originale.
- zeroShot_combined.csv: dataset ottenuto dalla concatenazione di zeroShot con dataset originale.
- train_StackoverFlowOriginale.csv: dataset train originale.
- test_StackoverFlowOriginali.csv: dataset test.

## Prerequisiti
Lo script può essere eseguito su CPU, ma per dataset più grandi o per molte ripetizioni è consigliata una GPU.

## Preparazione
Assicurati che i tuoi file CSV (train e test) abbiano almeno le seguenti colonne:
- ID: identificativo unico per ogni riga
- Polarity: etichetta del sentiment (positive, neutral, negative)
- Text: testo da classificare

## Esecuzione
'''bash

- chmod +x run_all_trainings.sh
- ./run_all_trainings.sh (oppure bash run_all_trainings.sh)

## Outputs
Cartella Predictions contenente i risultati dell'esecuzione (JSON e csv)
