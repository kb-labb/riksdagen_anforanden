import multiprocessing as mp
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.audio import split_audio_by_speech

df = pd.read_parquet("data/df_audio_metadata.parquet")
df_groups = df.groupby("dokid")
df_groups = df_groups[["dokid", "anforande_nummer", "filename", "start", "duration"]]
df_list = [df_groups.get_group(x) for x in df_groups.groups]  # list of dfs, one for each dokid

pool = mp.Pool(24)
df_dokids = pool.map(split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4)
pool.close()

df["filename_anforande_audio"] = df[["start", "duration", "filename"]].apply(
    lambda x: Path(x["filename"]).parent
    / f"{Path(x['filename']).stem}_{x['start']}_{x['start'] + x['duration']}.wav",
    axis=1,
)

df["filename_anforande_audio"] = df["filename_anforande_audio"].apply(lambda x: str(x))
df.to_parquet("data/df_audio_metadata.parquet", index=False)
