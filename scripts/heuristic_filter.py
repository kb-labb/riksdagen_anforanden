import os
import sys
import shutil
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from pyannote.audio import Pipeline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import split_audio_by_speech

pd.set_option("display.max_colwidth", 110)


df = pd.read_parquet("data/df_final.parquet")

df["start_diff"] = df["start_segment"] - df["start"]
df["end_diff"] = df["end_segment"] - df["end"]

# Use and keep official metadata for misdiarized speeches from 2013-01-01 and afterwards.
# We revert to using official metadata for 656 speeches (out of ~120k).
df_keep = df[(df["start_segment_same"]) & (~df["duplicate_speech"]) & (df["debatedate"] > "2013-01-01")]
df_keep_start = df[(df["start_diff"] < -5) & (df["start_diff"] > -55) & (df["debatedate"] > "2013-01-01")]

# Replace rows in df with rows in df_keep
df.loc[df_keep.index, ["start_segment", "end_segment", "duration_segment"]] = df_keep[
    ["start", "end", "duration"]
].values

df.loc[df_keep_start.index, "start_segment"] = df_keep_start["start"].values

df["is_prediction"] = True
df.loc[df_keep.index | df_keep_start.index, "is_prediction"] = False

# Quality filter for spechees
df = df[
    (df["duration_segment"] > 25)
    & ((df["end_text_time"] - df["start_text_time"]) > 15)
    & (df["length_ratio"] > 0.8)
    & (df["length_ratio"] < 1.5)
    & (df["overlap_ratio"] > 0.8)
    & (df["nr_speech_segments"] == 1)
    & (~(df["start_segment_diff"] < 5))
].reset_index(drop=True)


df["duration_segment"] = df["end_segment"] - df["start_segment"]
df["duration_segment"].sum() / 3600


df["start_adjusted"] = (df["start_segment"] - 0.1).round(1)
df["end_adjusted"] = (df["end_segment"] + 0.4).round(1)

df_pred = df[~df["is_prediction"]]


df_groups = df_pred.groupby("dokid")
df_groups = df_groups[["dokid", "anforande_nummer", "filename", "start", "duration", "start_adjusted", "end_adjusted"]]
df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid


# New speech audio file splits for the speeches whose start/end times were adjusted
pool = mp.Pool(16)
df_dokids = pool.map(split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=2)
pool.close()

df["filename_anforande_audio"] = df[["start_adjusted", "end_adjusted", "filename"]].apply(
    lambda x: Path(x["filename"]).parent / f"{Path(x['filename']).stem}_{x['start_adjusted']}_{x['end_adjusted']}.wav",
    axis=1,
)

df["filename_anforande_audio"] = df["filename_anforande_audio"].apply(lambda x: str(x))
# df.to_parquet("data/df_audio_metadata_final.parquet", index=False)

# Move wav files from data/audio2 to data/audio if they don't exist in data/audio
for filename in tqdm(df["filename_anforande_audio"].unique()):
    if (not (Path("data/audio") / Path(filename)).exists()) and (Path("data/audio2") / Path(filename)).exists():
        shutil.move((Path("data/audio2") / Path(filename)), (Path("data/audio") / Path(filename)))


df.to_parquet("data/df_final_final.parquet", index=False)

####
# GR01JUU18 klipptes inte rätt. Del av Peter Althins tal (164) klipptes bort.
# Anförande 165 finns inte, och klipptes förmodligen bort automatiskt.
# En massa anföranden saknas i denna debatt.
