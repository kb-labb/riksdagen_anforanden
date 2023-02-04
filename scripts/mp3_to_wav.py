import sys
import shutil
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm
from src.audio import convert_mp3_to_wav


df = pd.read_parquet("data/df_audio_metadata.parquet")

# Convert mp3 to wav using multiprocessing library mp.Pool
# Track progress with tqdm
with mp.Pool(16) as pool:
    pool.map(
        convert_mp3_to_wav,
        tqdm(df["filename"].unique(), total=len(df["filename"].unique())),
        chunksize=2,
    )


# df_file = df.groupby("dokid").first().reset_index()

# df_file["filename_wav"] = df_file["filename"].str.replace(".mp3", ".wav")

# # Check if filename_wav exists
# df_file["valid_wav"] = df_file["filename_wav"].apply(lambda x: (Path("data/audio") / Path(x)).exists())
# df_reprocess = df_file[(~df_file["valid_wav"])]

# with mp.Pool(7) as pool:
#     pool.map(
#         convert_mp3_to_wav,
#         tqdm(df_reprocess["filename"].unique(), total=len(df_reprocess["filename"].unique())),
#         chunksize=2,
#     )
