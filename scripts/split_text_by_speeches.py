import multiprocessing as mp
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from src.audio import split_text_by_speech

df = pd.read_parquet("data/df_final_riksvox.parquet")
df_groups = df.groupby("dokid")
df_list = [df_groups.get_group(x).copy() for x in df_groups.groups]  # list of dfs, one for each dokid

df_dokids = []
for df_dokid in tqdm(df_list, total=len(df_list)):
    df_dokids.append(split_text_by_speech(df_dokid))

df = pd.concat(df_dokids)
df["filename_anforande_text"] = df["filename_anforande_text"].apply(lambda x: str(x))


df.to_parquet("data/df_final_riksvox.parquet", index=False)
df[["filename_anforande_audio", "filename_anforande_text"]].to_csv(
    r"data/speeches_files_aeneas.txt", sep=" ", index=None, mode="a", header=None
)  # For aeneas
