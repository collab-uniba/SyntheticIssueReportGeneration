import pandas as pd
import argparse
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# PARSING ARGOMENTI
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="Path del dataset CSV"
)

parser.add_argument(
    "-o",
    "--original",
    type=str,
    required=False,
    help="Path del dataset originale CSV per confronto"
)

args = parser.parse_args()

# -------------------------
# LETTURA DATASET
# -------------------------
df = pd.read_csv(args.dataset, sep=';', quotechar='"')

# Normalizza i nomi delle colonne a lowercase per evitare KeyError
df.columns = df.columns.str.strip().str.lower()


print("Dataset caricato:", args.dataset)
print("Numero righe:", len(df))

results = {}

# -------------------------
# 1. VALORI MANCANTI indica la proporzione di campi vuoti o nulli nel dataset. Più è alto, più il dataset contiene rumore
# -------------------------
missing = df.isnull().sum().sum()
missing_ratio = missing / (df.shape[0] * df.shape[1])
results["missing_ratio"] = missing_ratio

# -------------------------
# 2. DISTRIBUZIONE CLASSI indica quanto sono bilanciate le classi nel dataset. Più è sbilanciata, più il dataset contiene rumore
# -------------------------
class_distribution = df["polarity"].value_counts(normalize=True)
imbalance = class_distribution.max() - class_distribution.min()
results["class_imbalance"] = imbalance

# -------------------------
# 3. DUPLICATI indica la proporzione di righe duplicate nel dataset. Più è alto, più il dataset contiene rumore
# -------------------------
duplicates = df["text"].duplicated().sum()
duplicate_ratio = duplicates / len(df)
results["duplicate_ratio"] = duplicate_ratio

# -------------------------
# 4. LUNGHEZZA TESTI indica la distribuzione delle lunghezze dei testi nel dataset. Testi troppo corti o troppo lunghi possono indicare rumore
# -------------------------
df["length"] = df["text"].apply(len)
length_stats = df["length"].describe()

# -------------------------
# 5. CARATTERI SPECIALI indica la proporzione di caratteri non alfanumerici nei testi del dataset. Più è alto, più i testi contengono rumore
# -------------------------
def count_special(text):
    return len(re.findall(r'[^a-zA-Z0-9\s]', str(text)))

df["special_chars"] = df["text"].apply(count_special)
special_ratio = df["special_chars"].mean()
results["special_char_noise"] = special_ratio

# -------------------------
# 6. NUMERI indica la proporzione di cifre nei testi del dataset. Più è alto, più i testi contengono rumore
# -------------------------
def count_digits(text):
    return sum(c.isdigit() for c in str(text))

df["digits"] = df["text"].apply(count_digits)
digit_ratio = df["digits"].mean()
results["digit_noise"] = digit_ratio

# -------------------------
# 7. INCOERENZA LABEL indica il numero di testi che hanno più di un'etichetta associata. Più è alto, più il dataset contiene rumore
# -------------------------
conflicts = df.groupby("text")["polarity"].nunique()
label_conflicts = (conflicts > 1).sum()
results["label_conflicts"] = label_conflicts

# -------------------------
# 8. VOCABOLARIO indica la dimensione del vocabolario unico presente nei testi del dataset. Più è alto, più il dataset è vario
# -------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
vocab_size = len(vectorizer.vocabulary_)
results["vocab_size"] = vocab_size

# -------------------------
# 9. TYPE TOKEN RATIO indica la proporzione tra il numero di parole uniche (tipi) e il numero totale di parole (token) nei testi del dataset. Più è alto, più il dataset è vario
# -------------------------
tokens = " ".join(df["text"]).split()
unique_tokens = set(tokens)
ttr = len(unique_tokens) / len(tokens)
results["type_token_ratio"] = ttr

# -------------------------
# 10. SIMILARITÀ TRA FRASI indica la proporzione di coppie di testi che sono molto simili tra loro. Più è alto, più il dataset contiene rumore
# -------------------------
sample = df["text"].sample(min(100, len(df)))
vectorizer = CountVectorizer().fit_transform(sample)
vectors = vectorizer.toarray()
similarity = cosine_similarity(vectors)
similar_pairs = np.sum(similarity > 0.9) - len(sample)
similar_ratio = similar_pairs / len(sample)
results["semantic_duplicates"] = similar_ratio

# -------------------------
# 11. SIMILARITÀ SINTETICO VS ORIGINALE
# -------------------------
if args.original:
    original = pd.read_csv(args.original, sep=';', quotechar='"')
    original.columns = original.columns.str.strip().str.lower()
    sample_size = min(100, len(df), len(original))
    synthetic_sample = df["text"].sample(sample_size, random_state=42)
    original_sample = original["text"].sample(sample_size, random_state=42)

    vectorizer = CountVectorizer().fit(pd.concat([synthetic_sample, original_sample]))
    synthetic_vectors = vectorizer.transform(synthetic_sample).toarray()
    original_vectors = vectorizer.transform(original_sample).toarray()

    similarity_matrix = cosine_similarity(synthetic_vectors, original_vectors)
    cross_similarity = similarity_matrix.mean()
    results["synthetic_original_similarity"] = round(cross_similarity, 4)

# -------------------------
# CALCOLO DATA QUALITY SCORE 
# -------------------------
score = 100
score = 100
score -= missing_ratio * 40
score -= duplicate_ratio * 20
score -= label_conflicts * 30
score -= special_ratio * 5
score -= digit_ratio * 5
score -= imbalance * 10
if "synthetic_original_similarity" in results:
    similarity_score = results["synthetic_original_similarity"] * 20
    score += similarity_score
results["data_quality_score"] = max(0, round(score,2))

print("\n===== DATA QUALITY SCORE =====")
print(results["data_quality_score"])

# -------------------------
# SALVATAGGIO REPORT
# -------------------------
report = pd.DataFrame([results])
report.to_csv("dataset_quality_report.csv", index=False)

print("\nReport salvato in dataset_quality_report.csv")