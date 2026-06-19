import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langfuse import Langfuse
from langfuse.experiment import LocalExperimentItem


langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_BASE_URL"]
)

# SEED (reproducibility)
np.random.seed(42)

# model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load data
def load_dataset(path):
    df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = ["ID", "Polarity", "Text"]
    df = df.dropna()
    df["Text"] = df["Text"].astype(str)
    return df

real = load_dataset("github.csv")

real_emb = model.encode(
    real["Text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

datasets = {
    "P1_creative": load_dataset("zeroShot_creative.csv"),
    "P2_balanced": load_dataset("zeroShot_balanced.csv"),
    "P3_strict": load_dataset("zeroShot_strict.csv")
}

# embeddings
real_emb = model.encode(
    real["Text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

# Core metrics
def compute_metrics(real_emb, synth_emb):

    sim_rs = cosine_similarity(synth_emb, real_emb)

    similarity = np.mean(np.max(sim_rs, axis=1))
    coverage = np.mean(np.max(sim_rs, axis=0))

    # intra-similarity (diversity proxy)
    if len(synth_emb) > 1:
        sim_ss = cosine_similarity(synth_emb)
        diversity = 1 - np.mean(sim_ss[np.triu_indices(len(synth_emb), k=1)])
        diversity = float(np.clip(diversity, 0.0, 1.0))
    else:
        diversity = 0.0

    novelty = 1 - similarity

    return {
        "similarity": float(similarity),
        "coverage": float(coverage),
        "diversity": float(diversity),
        "novelty": float(novelty)
    }
    
# experiment data
experiment_data = [
    LocalExperimentItem(
        id=name,
        input=df.to_dict(orient="records")
    )
    for name, df in datasets.items()
]

# task langfuse
def task(item):
    df = pd.DataFrame(item["input"])

    synth_emb = model.encode(
        df["Text"].tolist(),
        normalize_embeddings=True
    )

    return compute_metrics(real_emb, synth_emb)

# run experiment
result = langfuse.run_experiment(
    name="zeroShot_prompt_eval",
    run_name="run_1",
    data=experiment_data,
    task=task
)

# metrics table
rows = []

for name, df in datasets.items():
    synth_emb = model.encode(df["Text"].tolist(), normalize_embeddings=True)
    metrics = compute_metrics(real_emb, synth_emb)

    rows.append({
        "prompt": name,
        **metrics
    })

df_metrics = pd.DataFrame(rows)

metrics_dict = df_metrics.to_dict(orient="records")

with open("prompt_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
print("\n=== METRICS ===")
print(df_metrics)

# Distribution distance
def distribution_distance(a_emb, b_emb, sample=300):

    sample = min(sample, len(a_emb), len(b_emb))

    a = a_emb[np.random.choice(len(a_emb), sample, replace=False)]
    b = b_emb[np.random.choice(len(b_emb), sample, replace=False)]

    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)

    return np.linalg.norm(a_mean - b_mean)

keys = list(datasets.keys())

dist_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)

embeddings = {
    k: model.encode(v["Text"].tolist(), normalize_embeddings=True)
    for k, v in datasets.items()
}

for i in keys:
    for j in keys:
        dist_matrix.loc[i, j] = distribution_distance(
            embeddings[i],
            embeddings[j]
        )
        
dist_matrix = dist_matrix.astype(float)

dist_matrix.to_json("prompt_distance_matrix.json", orient="index", indent=2)

print("\n=== DISTANCE MATRIX ===")
print(dist_matrix)

# heatmap matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    dist_matrix.astype(float),
    annot=True,
    cmap="viridis",
    fmt=".3f",
    square=True
)

plt.title("Prompt Distance Matrix")
plt.tight_layout()

plt.savefig("prompt_distance_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Table
summary_table = df_metrics.copy()
summary_table = summary_table.set_index("prompt")

metric_cols = ["similarity", "coverage", "diversity", "novelty"]

summary_table_percent = summary_table.copy()
summary_table_percent[metric_cols] = (summary_table_percent[metric_cols] * 100).round(2)
summary_table_percent[metric_cols] = summary_table_percent[metric_cols].astype(str) + "%"

summary_table_percent = summary_table_percent.reset_index()

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis("off")

table = ax.table(
    cellText=summary_table_percent.values,
    colLabels=summary_table_percent.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Prompt Metrics (percent)", pad=20)
plt.tight_layout()

plt.savefig("prompt_metrics_table_percent.png", dpi=300, bbox_inches="tight")
plt.show()

# bar plot
def compute_real_similarity(embeddings, real_emb):
    results = {}

    for name, emb in embeddings.items():
        sim = np.max(cosine_similarity(emb, real_emb), axis=1)
        results[name] = np.mean(sim)

    return results


real_stats = compute_real_similarity(embeddings, real_emb)

labels = list(real_stats.keys())
means = [real_stats[k] for k in labels]

# colori diversi per ogni barra
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

plt.figure(figsize=(7,5))

bars = plt.bar(labels, means, color=colors)

plt.title("Average Similarity to Real Dataset")
plt.ylabel("Max cosine similarity (synthetic → real)")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.3)

# valore sopra ogni barra
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig("similarity_to_real_barplot.png", dpi=300)
plt.show()

langfuse.flush()
