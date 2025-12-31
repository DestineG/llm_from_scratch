import pandas as pd

df = pd.read_csv(
    "data/raw/wmt_zh_en_training_corpus/wmt_zh_en_training_corpus.csv",
    nrows=5
)
print(df)
print(df.columns)
