# Random Forest + TF-IDF

Questo progetto utilizza il modello Random Forest con features TF-IDF e n-grams per la classificazione del sentiment in testi brevi.
È progettato per essere eseguito facilmente tramite uno script Python (train_rf_tfidf.py) e uno script Bash (run_rf_tfidf.sh).

## Contenuto del Repository
- train_rf_tfidf.py: Script principale per il training, la valutazione e la predizione.
- run_rf_tfidf.sh: Script per gestire l'esecuzione.
- rf_tfidf_results/: cartella che contiene tutte le predizioni e le metriche generate dagli esperimenti.

## Preparazione
Assicurati che i tuoi file train.csv e test.csv abbiano almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_rf_tfidf.sh
- ./run_rf_tfidf.sh train_*.csv test_*.csv

## Outputs
- I file .csv e .json vengono salvati in: rf_tfidf_outputs/
