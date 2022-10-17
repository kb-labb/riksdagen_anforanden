import multiprocessing as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.audio import split_audio_by_speech

df = pd.read_parquet("data/df_audio_metadata_2020.parquet")
df_groups = df.groupby("dokid")
df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid


pool = mp.Pool(8)
df_dokids = pool.map(split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4)
pool.close()


df["filename_anforande_audio"] = df[["start", "duration", "filename"]].apply(
    lambda x: Path(x["filename"]).parent
    / f"{Path(x['filename']).stem}_{x['start']}_{x['start'] + x['duration']}.mp3",
    axis=1,
)

df["filename_anforande_audio"] = df["filename_anforande_audio"].apply(lambda x: str(x))
df.to_parquet("data/df_audio_metadata_2020.parquet", index=False)
