import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

df = pd.read_parquet("data/df_audio_metadata.parquet")
df = df[df["debatedate"] < "2010-01-01"].reset_index(drop=True)

pipe = pipeline(model="KBLab/wav2vec2-large-voxrex-swedish", device=0)

texts = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    try:
        audio_filepath = os.path.join("data/audio", row.filename_anforande_audio)
        text = pipe(audio_filepath, chunk_length_s=30, stride_length_s=5, return_timestamps="word")
        text["dokid"] = row.dokid
        text["anforande_nummer"] = row.anforande_nummer
        text["filename_anforande_audio"] = row.filename_anforande_audio
        texts.append(text)
    except Exception as e:
        print(e)
        print(row.filename_anforande_audio)
        texts.append(
            {
                "text": None,
                "dokid": row.dokid,
                "anforande_nummer": row.anforande_nummer,
                "filename_anforande_audio": row.filename_anforande_audio,
            }
        )

df_inference = pd.DataFrame(texts)
df_inference.to_parquet("data/df_inference_2003_2009.parquet")
