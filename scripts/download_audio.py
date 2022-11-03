import multiprocessing as mp
import pandas as pd
import sys
import argparse
from tqdm.contrib.concurrent import process_map
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import get_audio_file
from src.data import audio_to_dokid_folder, audiofile_exists

parser = argparse.ArgumentParser(
    description="""Read json files of riksdagens anföranden, save relevant metadata fields to file."""
)
parser.add_argument("-f", "--folder", type=str, default="data/audio")

args = parser.parse_args()

# Read data
df = pd.read_parquet("data/df_audio_metadata.parquet")

# Subset only speeches which we haven't already downloaded audio for
df["audiofile_exists"] = df[["dokid", "audiofileurl"]].apply(
    lambda x: audiofile_exists(x.dokid, x.audiofileurl.rsplit("/", 1)[1], folder=args.folder),
    axis=1,
)

df_undownloaded = df[df["audiofile_exists"] == False].reset_index(drop=True)
audio_download_urls = df_undownloaded["audiofileurl"].unique().tolist()

# Download audio files
process_map(get_audio_file, audio_download_urls, max_workers=mp.cpu_count(), chunksize=5)

# Move audio files to folder named after dokid
audio_to_dokid_folder(df)

# Add inferred filepaths to df (some files may not have been downloaded)
df["filename"] = df.apply(lambda x: x["dokid"] + "/" + x["audiofileurl"].rsplit("/", 1)[1], axis=1)
df = df.drop(columns=["audiofile_exists"])

# Save updated df with audio filepaths
df.to_parquet("data/df_audio_metadata.parquet", index=False)
