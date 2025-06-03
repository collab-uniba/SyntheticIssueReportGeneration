#  Analisi Few-Shot con SetFit: Performance in base alla quantità di dati

Questo script ha come scopo il valutare le prestazioni di SetFit su diversi sample per verificare dove otteniamo le prestazioni migliori (o per capire da che punto in poi le otteniamo) sulla metrica f1-score.
Lo script genererà a fine un grafico per rendere visibile il risultato dell'analisi.

---

## Contenuti del Repository

- `f1_score_vs_sample_size.png` — Grafico generato automaticamente.
- `requirements_grafico.txt` — Dipendenze Python per eseguire lo script.
- run_training_graf.sh
- plot_f1_vs_sample_size.py

---

## Requisiti

Python = 3.11

```bash
pip install -r requirements_grafico.txt

## Run Facile

Per utilizzare lo script si può utilizzare  run_training_graf.sh per semplificare il run.

Basterà usare il comando chmod +x run_training.sh la prima volta che si usa lo script e per poi usare ./run_training.sh train.csv test.csv per runnarlo senza problemi. Nel nostro caso scriveremo ./run_training.sh train_StackOverFlow.csv test_StackOverFlow.csv



