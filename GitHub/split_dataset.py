import pandas as pd
from sklearn.model_selection import train_test_split

# carica dataset
df = pd.read_csv("github_gold.csv", sep=";")

# split stratificato
df_train, df_test = train_test_split(
    df,
    test_size=0.3,              # 30% test
    stratify=df["Polarity"],    # mantiene distribuzione classi
    random_state=42             # riproducibile
)

# salva train
df_train.to_csv(
    "train_github_gold.csv",
    index=False,
    sep=";",
    encoding="utf-8"
)

# salva test
df_test.to_csv(
    "test_github_gold.csv",
    index=False,
    sep=";",
    encoding="utf-8"
)

print("Train size:", len(df_train))
print("Test size:", len(df_test))