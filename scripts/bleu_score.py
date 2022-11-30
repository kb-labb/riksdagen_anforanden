import sys
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import calculate_bleu, contiguous_ngram_match_star, contiguous_fuzzy_match_star
from src.data import normalize_text

df = pd.read_parquet("data/df_audio_metadata.parquet")
# df = df[(df["debatedate"] < "2019-01-01") | (df["debatedate"] >= "2020-01-01")].reset_index(
#     drop=True
# )

df_inference = pd.read_parquet("data/df_inference.parquet")
df_inference = df_inference.rename(columns={"text": "anftext_inference"})
df_inference.loc[df_inference["anftext_inference"] == "", "anftext_inference"] = None
df = pd.merge(df, df_inference, on=["dokid", "anforande_nummer", "filename_anforande_audio"])


df = normalize_text(df, column_in="anftext", column_out="anftext_normalized")
# Wav2vec2 inference should already be normalized, but sometimes contains multiple spaces
df = normalize_text(df, column_in="anftext_inference", column_out="anftext_inference")

df["bleu_score"] = df[["anftext_normalized", "anftext_inference"]].apply(
    lambda x: calculate_bleu(x.anftext_normalized, x.anftext_inference), axis=1
)

a = df[df["bleu_score"] > 0.3].reset_index(drop=True)

with mp.Pool() as pool:
    args = [
        (text1, text2, 6) for text1, text2 in zip(a["anftext_normalized"], a["anftext_inference"])
    ]
    a["contiguous_ngram_match"] = list(
        tqdm(
            pool.imap(
                contiguous_ngram_match_star,
                args,
                chunksize=1,
            ),
            total=len(a),
        )
    )


with mp.Pool() as pool:
    args = [
        (text1, text2, 55) for text1, text2 in zip(a["anftext_normalized"], a["anftext_inference"])
    ]
    a["contiguous_fuzzy_match"] = list(
        tqdm(
            pool.imap(
                contiguous_fuzzy_match_star,
                args,
                chunksize=1,
            ),
            total=len(a),
        )
    )


a[1580:1700].apply(
    lambda x: contiguous_fuzzy_match_star((x["anftext_normalized"], x["anftext_inference"], 55)),
    axis=1,
)


df["year"] = df["debatedate"].dt.year
print(df.groupby("year", as_index=False)["bleu_score"].mean().to_markdown(index=False))
df.groupby("year", as_index=False)["bleu_score"].median()


text1 = df[df["bleu_score"] > 0.2]["anftext_normalized"].iloc[1]
text2 = df[df["bleu_score"] > 0.2]["anftext_inference"].iloc[1]

text2 = df[(df["bleu_score"] < 0.1) & (df["bleu_score"] >= 0.05)]["anftext_inference"].iloc[203]
text1 = df[(df["bleu_score"] < 0.1) & (df["bleu_score"] >= 0.05)]["anftext_normalized"].iloc[203]
