import multiprocessing as mp
import pandas as pd
from tqdm.contrib.concurrent import process_map
from src.api import get_audio_file

df = pd.read_parquet("data/df_audio_metadata.parquet")
audio_download_urls = df["audiofileurl"].unique().tolist()

process_map(get_audio_file, audio_download_urls, max_workers=mp.cpu_count(), chunksize=5)

df["filename"] = df["audiofileurl"].apply(lambda link: link.rsplit("/", 1)[1])
df.to_parquet("data/df_audio_metadata.parquet", index=False)
