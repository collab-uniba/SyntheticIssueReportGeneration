import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione dataset e output
DATASET_PATH = "github_gold.csv"
OUTPUT_DIR = "dataset_exploration"

# Setup
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

print("=" * 60)
print("Analisi esplorativa del dataset")
print("=" * 60)

# Caricamento dataset
print("\n[1] Caricamento dataset...")

df = pd.read_csv(
    DATASET_PATH,
    sep=";",
    encoding="utf-8"
)

print(f"Dataset caricato: {DATASET_PATH}")

original_columns = df.columns.tolist()

# Informazioni generali
print("\n[2] Informazioni generali")

print(f"Numero di righe: {len(df)}")
print(f"Numero di colonne: {len(df.columns)}")

print("\nColonne:")
print(df.columns.tolist())

# Analisi dei valori mancanti
print("\n[3] Analisi dei valori mancanti")

missing_values = df.isnull().sum()

print(missing_values)

# Salva valori mancanti
missing_values.to_csv(output_path / "missing_values.csv")

# Duplicati
print("\n[4] Analisi dei duplicati")

duplicate_rows = df.duplicated().sum()
duplicate_texts = df.duplicated(subset=["Text"]).sum()

print(f"Righe duplicate: {duplicate_rows}")
print(f"Testi duplicati: {duplicate_texts}")

# Distribuzione delle classi
print("\n[5] Distribuzione delle classi")

class_counts = df["Polarity"].value_counts()
class_percentages = df["Polarity"].value_counts(normalize=True) * 100

distribution_df = pd.DataFrame({
    "count": class_counts,
    "percentuale": class_percentages.round(2)
})

print(distribution_df)

# Salva distribuzione delle classi
distribution_df.to_csv(output_path / "class_distribution.csv")

# Grafico distribuzione delle classi
plt.figure(figsize=(8, 5))

class_counts.plot(kind="bar")

plt.title("Distribuzione delle classi")
plt.xlabel("Polarity")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(output_path / "class_distribution.png")
plt.close()

# Analisi della lunghezza del testo
print("\n[6] Analisi della lunghezza del testo")

# Converti Text in stringa e gestisci valori mancanti
df["Text"] = df["Text"].fillna("").astype(str)

# Lunghezza in caratteri
df["text_length_chars"] = df["Text"].apply(len)

# Lunghezza in parole
df["text_length_words"] = df["Text"].apply(lambda x: len(x.split()))

text_stats = {
    "media_caratteri": round(df["text_length_chars"].mean(), 2),
    "mediana_caratteri": round(df["text_length_chars"].median(), 2),
    "min_caratteri": int(df["text_length_chars"].min()),
    "max_caratteri": int(df["text_length_chars"].max()),
    "media_parole": round(df["text_length_words"].mean(), 2),
    "mediana_parole": round(df["text_length_words"].median(), 2),
    "min_parole_per_frase": int(df["text_length_words"].min()),
    "max_parole_per_frase": int(df["text_length_words"].max()),
}

for key, value in text_stats.items():
    print(f"{key}: {value}")

# Salva statistiche sulla lunghezza del testo
pd.DataFrame([text_stats]).to_csv(
    output_path / "text_length_statistics.csv",
    index=False
)

# Grafico distribuzione lunghezza del testo (caratteri)
plt.figure(figsize=(10, 6))

plt.hist(df["text_length_chars"], bins=50)

plt.title("Distribuzione della lunghezza del testo (Caratteri)")
plt.xlabel("Caratteri")
plt.ylabel("Frequenza")

plt.tight_layout()

plt.savefig(output_path / "text_length_distribution_chars.png")
plt.close()

# Grafico distribuzione lunghezza del testo (parole)
plt.figure(figsize=(10, 6))

plt.hist(df["text_length_words"], bins=50)

plt.title("Distribuzione della lunghezza del testo (Parole)")
plt.xlabel("Parole")
plt.ylabel("Frequenza")

plt.tight_layout()

plt.savefig(output_path / "text_length_distribution_words.png")
plt.close()

# Samples casuali
print("\n[7] Esempi casuali")

sample_df = df.sample(min(10, len(df)), random_state=42)

print(sample_df[["Polarity", "Text"]])

sample_df.to_csv(
    output_path / "random_samples.csv",
    index=False,
    encoding="utf-8"
)

# Lunghezza media del testo per classe
print("\n[8] Lunghezza media del testo per classe")

class_length_stats = df.groupby("Polarity")["text_length_chars"].mean().round(2)

print(class_length_stats)

class_length_stats.to_csv(
    output_path / "classwise_text_length.csv"
)

# Grafico lunghezza media del testo per classe
plt.figure(figsize=(8, 5))

class_length_stats.plot(kind="bar")

plt.title("Lunghezza media del testo per classe")
plt.xlabel("Polarity")
plt.ylabel("Lunghezza media (caratteri)")

plt.tight_layout()

plt.savefig(output_path / "classwise_text_length.png")
plt.close()


# Report finale
print("\n[9] Salvataggio report finale")

summary_report = f"""
REPORT ANALISI ESPLORATIVA
==========================

Dataset: {DATASET_PATH}

INFORMAZIONI GENERALI
-------
Righe: {len(df)}
Colonne: {len(original_columns)}

VALORI MANCANTI
--------------
{missing_values.to_string()}

DUPLICATI
----------
Righe duplicate: {duplicate_rows}
Testi duplicati: {duplicate_texts}

DISTRIBUZIONE DELLE CLASSI
------------------
{distribution_df.to_string()}

STATISTICHE LUNGHEZZA TESTO
----------------------
{pd.DataFrame([text_stats]).to_string(index=False)}

LUNGHEZZA MEDIA TESTO PER CLASSE
----------------------------
{class_length_stats.to_string()}
"""

with open(output_path / "summary_report.txt", "w", encoding="utf-8") as f:
    f.write(summary_report)

# Stampa report finale
print("\n" + "=" * 60)
print("Analisi esplorativa completata.")
print(f"Risultati salvati in: {OUTPUT_DIR}")
print("=" * 60)