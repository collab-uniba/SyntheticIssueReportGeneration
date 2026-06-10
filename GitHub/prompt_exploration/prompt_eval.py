import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langfuse import Langfuse
from langfuse.experiment import LocalExperimentItem

# =========================
# INIT
# =========================
langfuse = Langfuse()
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD DATA
# =========================
def load_dataset(path):
    df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = ["ID", "Polarity", "Text"]

    df = df.dropna()
    df["Text"] = df["Text"].astype(str)
    return df


real = load_dataset("train_github.csv")

synthetic_a = load_dataset("train_fewShot_creative.csv")
synthetic_b = load_dataset("train_fewShot_balanced.csv")
synthetic_c = load_dataset("train_fewShot_strict.csv")

datasets = {
    "prompt_creative": synthetic_a,
    "prompt_balanced": synthetic_b,
    "prompt_strict": synthetic_c
}

# =========================
# REAL EMBEDDINGS
# =========================
real_emb = model.encode(real["Text"].tolist(), normalize_embeddings=True)
real_centroid = np.mean(real_emb, axis=0)

# =========================
# METRICS
# =========================
def compute_metrics(df, real_df):
    texts = df["Text"].tolist()

    synth_emb = model.encode(texts, normalize_embeddings=True)
    real_emb_local = model.encode(real_df["Text"].tolist(), normalize_embeddings=True)
    centroid = np.mean(synth_emb, axis=0)
    real_centroid = np.mean(real_emb_local, axis=0)

    similarity = cosine_similarity([centroid], [real_centroid])[0][0]

    if len(texts) > 1:
        sim_matrix = cosine_similarity(synth_emb)
        upper = sim_matrix[np.triu_indices(len(texts), k=1)]
        diversity = 1 - np.mean(upper)
    else:
        diversity = 0.0
        
    coverage = compute_coverage(real_emb_local, synth_emb)
    novelty = compute_novelty(real_emb_local, synth_emb)
    class_distance = compute_class_distance(real_df, df)

    return {
        "similarity": float(similarity),
        "diversity": float(diversity),
        "coverage": float(coverage),
        "novelty": float(novelty),
        "class_distance": float(class_distance)
    }

def compute_coverage(real_emb, synth_emb):
    # per ogni embedding reale, trova il più simile nel sintetico
    sim_matrix = cosine_similarity(real_emb, synth_emb)
    max_sim = np.max(sim_matrix, axis=1)
    return float(np.mean(max_sim))


def compute_novelty(real_emb, synth_emb):
    # per ogni sintetico, quanto è distante dal reale
    sim_matrix = cosine_similarity(synth_emb, real_emb)
    max_sim = np.max(sim_matrix, axis=1)
    return float(1 - np.mean(max_sim))

def compute_class_distance(real_df, synth_df):
    real_dist = real_df["Polarity"].value_counts(normalize=True)
    synth_dist = synth_df["Polarity"].value_counts(normalize=True)

    all_classes = set(real_dist.index).union(set(synth_dist.index))

    distance = 0.0
    for c in all_classes:
        p_real = real_dist.get(c, 0.0)
        p_synth = synth_dist.get(c, 0.0)
        distance += abs(p_real - p_synth)

    return float(distance / len(all_classes))

# =========================
# TASK (LANGFUSE)
# =========================
def task(item):
    df = pd.DataFrame(item["input"])
    metrics = compute_metrics(df, real)

    return {
        "prompt": item["id"],
        "similarity": metrics["similarity"],
        "diversity": metrics["diversity"],
        "coverage": metrics["coverage"],
        "novelty": metrics["novelty"],
        "class_distance": metrics["class_distance"]
    }

# =========================
# EVALUATORS
# =========================
def eval_similarity(**kwargs):
    return kwargs.get("output", {}).get("similarity", 0.0)

def eval_diversity(**kwargs):
    return kwargs.get("output", {}).get("diversity", 0.0)

def eval_coverage(**kwargs):
    return kwargs.get("output", {}).get("coverage", 0.0)

def eval_novelty(**kwargs):
    return kwargs.get("output", {}).get("novelty", 0.0)

def eval_class_distance(**kwargs):
    return kwargs.get("output", {}).get("class_distance", 0.0)


# =========================
# EXPERIMENT DATA
# =========================
experiment_data = [
    LocalExperimentItem(
        id=name,
        input=df.to_dict(orient="records")
    )
    for name, df in datasets.items()
]

# =========================
# RUN EXPERIMENT
# =========================
result = langfuse.run_experiment(
    name="prompt_comparison_experiment",
    run_name="run_1",
    data=experiment_data,
    task=task,
    evaluators=[eval_similarity, eval_diversity, eval_coverage, eval_novelty, eval_class_distance],
    metadata={
        "model": "llama3.2:1b",
        "task": "synthetic_dataset_evaluation"
    }
)

print(result)

# =========================
# EXPORT RAW RESULTS
# =========================
rows = []
for item in experiment_data:
    df = pd.DataFrame(item["input"])
    metrics = compute_metrics(df, real)

    rows.append({
        "prompt": item["id"],
        "similarity": metrics["similarity"],
        "diversity": metrics["diversity"],
        "coverage": metrics["coverage"],
        "novelty": metrics["novelty"],
        "class_distance": metrics["class_distance"]
    })

with open("experiment_raw_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

# =========================
# SUMMARY + RANKINGS (LIGHTWEIGHT)
# =========================
df = pd.DataFrame(rows)

summary = df.groupby("prompt").agg(
    similarity_mean=("similarity", "mean"),
    diversity_mean=("diversity", "mean"),
    coverage_mean=("coverage", "mean"),
    novelty_mean=("novelty", "mean"),
    class_distance_mean=("class_distance", "mean")
).reset_index()

rankings = {
    "ranking_similarity": df.groupby("prompt")["similarity"].mean().sort_values(ascending=False).to_dict(),
    "ranking_diversity": df.groupby("prompt")["diversity"].mean().sort_values(ascending=False).to_dict(),
    "ranking_coverage": df.groupby("prompt")["coverage"].mean().sort_values(ascending=False).to_dict(),
    "ranking_novelty": df.groupby("prompt")["novelty"].mean().sort_values(ascending=False).to_dict(),
    "ranking_class_distance": df.groupby("prompt")["class_distance"].mean().sort_values(ascending=False).to_dict()
}

with open("experiment_rankings.json", "w", encoding="utf-8") as f:
    json.dump(rankings, f, indent=2, ensure_ascii=False)

# =========================
# SINGLE TESI-READY PLOT
# =========================
plot_df = summary.set_index("prompt")

ax = plot_df[["similarity_mean", "diversity_mean"]].plot(kind="bar")
ax.set_title("Prompt comparison: similarity vs diversity")
ax.set_ylabel("score")
ax.set_xlabel("prompt")

plt.xticks(rotation=0)

# =========================
# VALUE LABELS ON TOP
# =========================
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=9
    )

plt.ylim(0, 1)  # utile per leggibilità (similarity/diversity sono 0-1)

plt.tight_layout()
plt.savefig("prompt_comparison_chart.png", dpi=300)
plt.show()

# TABELLA

summary_table = plot_df[["similarity_mean", "diversity_mean", "coverage_mean", "novelty_mean", "class_distance_mean"]].copy()

summary_table = summary_table.reset_index()

summary_table.columns = ["prompt", "similarity", "diversity", "coverage", "novelty", "class_distance"]

# coversione percentuale
summary_table["similarity"] = (summary_table["similarity"] * 100).round(2).astype(str) + "%"
summary_table["diversity"] = (summary_table["diversity"] * 100).round(2).astype(str) + "%"
summary_table["coverage"] = (summary_table["coverage"] * 100).round(2).astype(str) + "%"
summary_table["novelty"] = (summary_table["novelty"] * 100).round(2).astype(str) + "%"
summary_table["class_distance"] = (summary_table["class_distance"] * 100).round(2).astype(str) + "%"

fig, ax = plt.subplots(figsize=(8, 2))

ax.axis("off")  # nasconde assi

table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Summary Table (percent values)", pad=20)

plt.tight_layout()
plt.savefig("experiment_summary_table.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== SUMMARY TABLE ===")
print(summary_table)

# =========================
# RADAR PREPARATION
# =========================

metrics = ["similarity_mean", "diversity_mean", "coverage_mean", "novelty_mean", "class_distance_mean"]

radar_df = summary.set_index("prompt")[metrics]

# normalizzazione 0-1 (importante per confronto corretto)
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

labels = ["Similarity", "Diversity", "Coverage", "Novelty", "Class Distance"]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # chiusura cerchio

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for prompt in radar_norm.index:
    values = radar_norm.loc[prompt].tolist()
    values += values[:1]  # chiusura

    ax.plot(angles, values, linewidth=2, label=prompt)
    ax.fill(angles, values, alpha=0.1)
    
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_title("Synthetic Prompt Evaluation Radar", pad=20)

ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("radar_prompt_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# DONE
# =========================
print("\n✔ DONE")
print("Generated files:")
print("- experiment_raw_results.json")
print("- experiment_rankings.json")
print("- experiment_summary_table.png")
print("- prompt_comparison_chart.png")
print("- radar_prompt_comparison.png")