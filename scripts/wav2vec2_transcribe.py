import sys
import os
from pathlib import Path
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import transcribe

df = pd.read_parquet("data/df_final_metadata.parquet")
df = df[df["debatedate"] < "2016-01-01"]

pipe = pipeline(model="KBLab/wav2vec2-large-voxrex-swedish", device=0)

df_inference = transcribe(df, full_debate=True)

df_inference = df_inference.rename(columns={"text": "anftext_inference"})
df_inference.loc[df_inference["anftext_inference"] == "", "anftext_inference"] = None
df_inference.to_parquet("data/df_inference_eval_2003_2016.parquet")
