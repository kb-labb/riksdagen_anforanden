import multiprocessing as mp
import pandas as pd
import sys
from tqdm.contrib.concurrent import process_map
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import get_audio_file
from src.data import audio_to_dokid_folder

df = pd.read_parquet("data/df_audio_metadata.parquet")
audio_download_urls = df["audiofileurl"].unique().tolist()

process_map(get_audio_file, audio_download_urls, max_workers=mp.cpu_count(), chunksize=5)

audio_to_dokid_folder(df)  # Move audio files to folder named after dokid
df["filename"] = df.apply(lambda x: x["dokid"] + "/" + x["audiofileurl"].rsplit("/", 1)[1], axis=1)

df.to_parquet("data/df_audio_metadata.parquet", index=False)
