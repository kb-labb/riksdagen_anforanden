import sys
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import split_audio_by_speech, get_corrupt_audio_files

"""
Attempt to correct corrupt audio files by re-processing them.

Run this script after force_align_audio.py if you get errors about corrupt audio files.
"""

df = pd.read_parquet("data/df_audio_metadata.parquet")
df = get_corrupt_audio_files(df)

df_groups = df.groupby("dokid")
df_list = [
    df_groups.get_group(x).copy() for x in df_groups.groups
]  # list of dfs, one for each dokid

pool = mp.Pool(20)
df_dokids = pool.map(split_audio_by_speech, tqdm(df_list, total=len(df_list)), chunksize=4)
pool.close()
