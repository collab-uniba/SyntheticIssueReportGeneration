#  Classificazione con SetFit

Questo progetto utilizza **SetFit** (Sentence Transformer Fine-tuning) per la classificazione del sentiment in testi brevi.
È progettato per essere eseguito facilmente tramite uno script Python (`train_model.py`) e uno script Bash (`train_and_run.sh`).
Supporta anche il training few-shot specificando quanti esempi usare per classe.

---

##  Contenuti del Repository

- `train_model.py` — Script principale per il training e la predizione.
- `train_and_run.sh` — Script shell per gestire l'ambiente virtuale, le dipendenze e l'esecuzione.
- `requirements.txt` — Lista delle dipendenze necessarie.
- `test_predictions.csv` — File CSV generato con le predizioni sul test set.

---

## Requisiti

- Python 3.11
- I pacchetti necessari vengono installati automaticamente da `requirements.txt`.

## Dettagli dei Parametri
-d, --train_file (stringa, obbligatorio)
Percorso al file CSV contenente i dati di training. Deve includere almeno le colonne Text e Polarity.

-t, --test_file (stringa, opzionale)
Percorso al file CSV contenente i dati di test. Deve includere almeno la colonna Text. Se non fornito, il dataset di training verrà suddiviso automaticamente in base al parametro split_ratio.

-n, --num_samples (intero, opzionale)
Numero di esempi per etichetta da usare per il training. Se impostato a 0, verrà utilizzato l'intero dataset. Utile per il training few-shot.

-s, --split_ratio (float, opzionale)
Percentuale del dataset di training da usare come test se non è fornito un test set. Deve essere un valore compreso tra 0 e 1. Default: 0.3.

## Esecuzione Rapida

### 1. Prepara i tuoi file CSV

Assicurati che i tuoi file `train.csv` e (opzionalmente) `test.csv` abbiano almeno le colonne:

- `ID`
- `Text`
- `Polarity` (valori: `positive`, `negative`, `neutral`)



