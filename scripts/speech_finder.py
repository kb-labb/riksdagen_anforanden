import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from torch.utils.data import DataLoader
from transformers import pipeline
from tqdm import tqdm
from src.metrics import (
    contiguous_ngram_indices,
    contiguous_fuzzy_indices,
)
from src.data import normalize_text
from src.dataset import DiarizationDataset
from src.audio import diarize, transcribe
from pyannote.audio import Pipeline

# Read df_audiometa.parquet
df_audiometa = pd.read_parquet("data/df_audio_metadata.parquet")

pipe = pipeline(model="KBLab/wav2vec2-large-voxrex-swedish", device=0)

df_debates = df_audiometa.groupby("dokid").first().reset_index()
df_debates = df_debates[df_debates["debatedate"] < "2017-01-01"]


#### TRANSCRIBE ####
df_inference = transcribe(
    df_debates,
    pipe,
    folder="data/audio",
    chunk_length_s=50,
    stride_length_s=7,
    return_timestamps="word",
    full_debate=True,
)

df_inference = df_inference.rename(columns={"text": "anftext_inference"})
df_inference.loc[df_inference["anftext_inference"] == "", "anftext_inference"] = None
df_inference.to_parquet("data/df_finder_2003_2016.parquet", index=False)


#### Fuzzy Timestamps ####
df_inference = pd.read_parquet("data/df_finder_2003_2016.parquet")
# Wav2vec2 inference should already be normalized, but sometimes contains multiple spaces
df_inference = normalize_text(df_inference, column_in="anftext_inference", column_out="anftext_inference")

df = df_audiometa[df_audiometa["dokid"].isin(df_debates["dokid"])].reset_index(drop=True)

# Join the columns anftext_inference and chunks from df_inference
df = df.merge(df_inference[["dokid", "anftext_inference", "chunks"]], on=["dokid"], how="left")

df = normalize_text(df, column_in="anftext", column_out="anftext_normalized")

df["match_indices_fuzzy"] = contiguous_fuzzy_indices(
    df, column_in="anftext_normalized", column_out="anftext_inference", threshold=55
)

df["match_indices_ngram"] = contiguous_ngram_indices(
    df,
    column_in="anftext_normalized",
    column_out="anftext_inference",
    n=6,
    threshold=1.8,
    min_continuous_match=13,
    max_gap=60,
    processes=24,
)

df["start_word_index"] = df["match_indices_fuzzy"].apply(lambda x: x[0]).astype("Int64")
df["end_word_index"] = df["match_indices_fuzzy"].apply(lambda x: x[1]).astype("Int64")
df["fuzzy_score"] = df["match_indices_fuzzy"].apply(lambda x: x[2])


def get_text_time(row, start=True):
    if start:
        if pd.notnull(row["start_word_index"]):
            return row["chunks"][row["start_word_index"]]["timestamp"][0]
        else:
            return None
    else:
        if pd.notnull(row["end_word_index"]):
            try:
                return row["chunks"][row["end_word_index"] - 1]["timestamp"][1]
            except IndexError:
                return row["chunks"][row["end_word_index"] - 2]["timestamp"][1]
        else:
            return None


df["start_text_time"] = df[["start_word_index", "chunks"]].apply(lambda x: get_text_time(x, start=True), axis=1)
df["end_text_time"] = df[["end_word_index", "chunks"]].apply(lambda x: get_text_time(x, start=False), axis=1)

df["end"] = df["start"] + df["duration"]
df["start_diff"] = df["start_text_time"] - df["start"]
df["end_diff"] = df["end_text_time"] - df["end"]

df[
    [
        "dokid",
        "anforande_nummer",
        "start",
        "end",
        "duration",
        "start_word_index",
        "end_word_index",
        "fuzzy_score",
        "start_text_time",
        "end_text_time",
        "start_diff",
        "end_diff",
    ]
].to_parquet("data/df_timestamp_2003_2016.parquet", index=False)


#### Diarization ####
df_timestamp = pd.read_parquet("data/df_timestamp_2003_2016.parquet")

# Left join the columns dokid, anforance_nummer, filename from df in to df_timestamp
df_timestamp = df_timestamp.merge(
    df[["dokid", "anforande_nummer", "filename", "debatedate", "valid_audio"]],
    on=["dokid", "anforande_nummer"],
    how="left",
)

df_timestamp["filename"] = df_timestamp["filename"].str.replace(".mp3", ".wav")
df_timestamp["valid_wav"] = df_timestamp["filename"].apply(
    lambda x: (Path("data/audio") / Path(x)).exists() if pd.notnull(x) else False
)

df_timestamp.loc[df_timestamp["valid_audio"].isna(), "valid_audio"] = False
df_timestamp = df_timestamp[(df_timestamp["valid_wav"])].reset_index(drop=True)


diarization = DiarizationDataset(df_timestamp, full_debate=True, folder="data/audio")


df_speakers = diarize(pipe=pipe, diarization_dataset=diarization)
df_speakers.to_parquet("data/df_speakers_debate_2003_2016.parquet", index=False)
