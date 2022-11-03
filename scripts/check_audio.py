import os
import pandas as pd
import numpy as np
from pydub import AudioSegment

df = pd.read_parquet("data/df_audio_metadata.parquet")

# df[df["audiofileurl"].str.contains("2442209030025807721_aud.mp3")]["debateurl"].iloc[0]
# df[df["audiofileurl"].str.contains("2442209030025807721_aud.mp3")]
# # Beatrice Ask i ljudfilen.
# df[df["anftext"].str.contains("genomdriva den största satsningen")]


# df[df["audiofileurl"].str.contains("2442210070028068221_aud.mp3")]
# df[df["audiofileurl"].str.contains("2442210070028068221_aud.mp3")]["debateurl"].iloc[0]
# df[df["anftext"].str.contains("realistiskt att helt ta bort fastighetsskatten")].iloc[:, 5:10]


df_sub = df.groupby(["dokid", "audiofileurl"]).first().reset_index()
files = os.listdir("data/audio")

df_sub.columns

df_sub["filename"] = df_sub["audiofileurl"].apply(lambda x: x.rsplit("/")[-1])
df_sub[df_sub["filename"].isin(files)]

sound = AudioSegment.from_mp3("data/audio/2442209030025807721_aud.mp3")

len(sound) / 1000 / 60


df = pd.read_parquet("data/df_anforanden_metadata.parquet")
df = df[~pd.isna(df["rel_dok_id"])].reset_index(drop=True)
df["anforande_nummer"] = df["anforande_nummer"].astype(int)

df_audio = pd.read_parquet("data/df_audio_metadata.parquet")

# Riksdagen is not consistent wither upper/lower case in ids
df_audio["rel_dok_id"] = df_audio["rel_dok_id"].str.upper()
df["rel_dok_id"] = df["rel_dok_id"].str.upper()
df_audio = df_audio.rename(columns={"number": "anforande_nummer"})
df_audio["dokid"] = df_audio["dokid"].str.upper()

df_audio = df_audio.merge(
    df[["rel_dok_id", "number", "anforandetext"]],
    left_on=["rel_dok_id", "anforande_number"],
    right_on=["rel_dok_id", "anforande_number"],
    how="left",
)

df_audio.loc[df_audio["anftext"] == "", "anftext"] = None
df_audio["anftext"] = df_audio["anftext"].fillna(df_audio["anforandetext"])


df_audio[df_audio["anftext"].isna()]


df_audio[df_audio["anforandetext"].isna()]
df_audio[df_audio["rel_dok_id"] == "H901KRU5"]
df[df["rel_dok_id"] == "H901KRU5"]

df[df["anforandetext"].isna()]
df_audio[df_audio["anforandetext"].isna()]

df_audio[df_audio["dokid"] == "GS01UBU6"]

# Missing text in both metadata files.
df_audio[(df_audio["anftext"] == "") & (df_audio["anforandetext"].isna())]

df[df["rel_dok_id"].str.contains("GS01UBU6")]

df[df["rel_dok_id"].str.contains(",")]
df_audio[df_audio["dokid"].str.contains(",")]


df_audio["anftext"]
