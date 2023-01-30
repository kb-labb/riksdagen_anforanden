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

df = df[df["bleu_score"] > 0.05].reset_index(drop=True)
# filter out those anftext_inference that are shorter than 7 words
df = df[
    (df["anftext_inference"].str.split().str.len() >= 8)
    & (df["anftext_normalized"].str.split().str.len() >= 8)
].reset_index(drop=True)

df = df[~df["anftext_normalized"].isna()].reset_index(drop=True)


def contiguous_text_indices_ngram(
    df,
    column_in,
    column_out,
    n=6,
    threshold=1.3,
    min_continuous_match=8,
    max_gap=30,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column name of text to match against.
        column_out (str): Column name of text we get matching indices for.
            n (int): N-gram sizes 1 to n.
        threshold (float): Threshold score for contiguous n-gram match to be considered a match.
        min_continous_match (int): Minimum continuous word matches for the region to
            be considered contiguous.
        max_gap (int): Maximum gap (in words) between contiguous region and the next/previous
            region for it to be seen as the start/end index of a larger joined together contiguous
            region.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    with mp.Pool(processes) as pool:
        args = [
            (text1, text2, n, threshold, min_continuous_match, max_gap)
            for text1, text2 in zip(df[column_in], df[column_out])
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


def contiguous_text_indices_fuzzy(
    df,
    column_in,
    column_out,
    threshold_fuzzy=55,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column name of text to match against.
        column_out (str): Column name of text we get matching indices for.
            n (int): N-gram sizes 1 to n.
        max_non_match (int): Maximum number of contigous indices below threshold score to allow
            for matching regions to no longer be considered contiguous.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    with mp.Pool(processes) as pool:
        args = [
            (text1, text2, threshold_fuzzy) for text1, text2 in zip(df[column_in], df[column_out])
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


df_inference_bleu = contiguous_text_indices_ngram(df, "anftext_normalized", "anftext_inference")
df_inference_bleu = contiguous_text_indices_fuzzy(df, "anftext_normalized", "anftext_inference")
df_anftext_bleu = contiguous_text_indices_ngram(df, "anftext_inference", "anftext_normalized")
df_anftext_bleu = contiguous_text_indices_fuzzy(df, "anftext_inference", "anftext_normalized")

df_inference_bleu.to_parquet("data/df_inference_bleu.parquet")
df_anftext_bleu.to_parquet("data/df_anftext_bleu.parquet")
