import sys
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import calculate_bleu, contiguous_ngram_match_star, contiguous_fuzzy_match_star
from src.data import normalize_text

df = pd.read_parquet("data/df_audio_metadata.parquet")
df_inference = pd.read_parquet("data/df_inference.parquet")

df = pd.merge(df, df_inference, on=["dokid", "anforande_nummer", "filename_anforande_audio"])

df = normalize_text(df, column_in="anftext", column_out="anftext_normalized")
# Wav2vec2 inference should already be normalized, but sometimes contains multiple spaces
df = normalize_text(df, column_in="anftext_inference", column_out="anftext_inference")

df["bleu_score"] = df[["anftext_normalized", "anftext_inference"]].apply(
    lambda x: calculate_bleu(x.anftext_normalized, x.anftext_inference), axis=1
)

df = df[df["bleu_score"] > 0.1].reset_index(drop=True)
# filter out those anftext_inference that are shorter than 7 words
df = df[
    (df["anftext_inference"].str.split().str.len() >= 8)
    & (df["anftext_inference"].str.split().str.len() >= 8)
].reset_index(drop=True)

# with mp.Pool() as pool:
#     args = [
#         (text1, text2, 6, 1.3, 9)
#         for text1, text2 in zip(df["anftext_normalized"], df["anftext_inference"])
#     ]
#     df["contiguous_ngram_match"] = list(
#         tqdm(
#             pool.imap(
#                 contiguous_ngram_match_star,
#                 args,
#                 chunksize=1,
#             ),
#             total=len(df),
#         )
#     )


# with mp.Pool() as pool:
#     args = [
#         (text1, text2, 55)
#         for text1, text2 in zip(df["anftext_normalized"], df["anftext_inference"])
#     ]
#     df["contiguous_fuzzy_match"] = list(
#         tqdm(
#             pool.imap(
#                 contiguous_fuzzy_match_star,
#                 args,
#                 chunksize=1,
#             ),
#             total=len(df),
#         )
#     )


# Anftext
with mp.Pool() as pool:
    args = [
        (text1, text2, 6, 1.3, 9)
        for text1, text2 in zip(df["anftext_inference"], df["anftext_normalized"])
    ]
    df["contiguous_ngram_match"] = list(
        tqdm(
            pool.imap(
                contiguous_ngram_match_star,
                args,
                chunksize=1,
            ),
            total=len(df),
        )
    )


with mp.Pool() as pool:
    args = [
        (text1, text2, 55)
        for text1, text2 in zip(df["anftext_inference"], df["anftext_normalized"])
    ]
    df["contiguous_fuzzy_match"] = list(
        tqdm(
            pool.imap(
                contiguous_fuzzy_match_star,
                args,
                chunksize=1,
            ),
            total=len(df),
        )
    )


df.to_parquet("data/df_anftext_bleu.parquet")
