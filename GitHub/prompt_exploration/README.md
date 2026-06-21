# Prompt evaluation

Questa cartella contiene ciò che è stato usato per effettuare la prompt evaluation su tre prompt differenti: uno bilaciato, uno creativo, uno rigido. L'analisi è stata fatta tramite lo script prompt_eval.py. una versione ridimesionata del dataset github originale viene confrontata con i dataset fewshot e zeroshot generati utilizzando i tre prompt differenti.
Sia il dataset originale che i dataset sintetici sono composti da 300 esempi per classe.

## Contenuto
- prompt_eval.py ---> script di analisi
- githyb.csv ---> script originare ridimensionato
- zeroShot/ ---> cartella che contiene i tre dataset zeroShot (zeroShot_balanced.csv, zeroShot_creative.csv, zeroShot_strict.csv). La cartella contiene anche grafici e risultati dell'analisi effettuata
- fewShot/ ---> cartella che contiene i tre dataset fewoShot (fewoShot_balanced.csv, fewoShot_creative.csv, fewoShot_strict.csv). La cartella contiene anche grafici e risultati dell'analisi effettuata
- prompt/ ---> cartella che contiene i tre prompt usati per la generazione dei dati sintetici (prompt_balanced.yaml, prompt_creative.yaml, prompt_strict.yaml)