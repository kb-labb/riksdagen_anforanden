import os
import pandas as pd
from pydub import AudioSegment
from nltk import sent_tokenize

df = pd.read_parquet("data/df_audio_metadata.parquet")
df_sub = df[df["audiofileurl"].str.contains("2442209030025807721_aud.mp3")].reset_index(drop=True)

segments = df_sub[["start", "duration"]].to_dict(orient="records")
sound = AudioSegment.from_mp3("2442209030025807721_aud.mp3")

os.makedirs("data/test", exist_ok=True)

for segment in segments:
    split = sound[
        (float(segment["start"]) * 1000) : (
            float(segment["start"]) + float(segment["duration"]) * 1000
        )
    ]
    split.export(f"data/test/test_{segment['start']}.mp3", format="mp3")

df_sub["lines"] = df_sub["anftext"].apply(lambda x: sent_tokenize(x))

for i in range(len(df_sub)):
    with open(f"data/text/anf{i}.txt", "w") as f:
        for line in df_sub["lines"][i]:
            f.write(line + "\n")


df = pd.read_json("map4.json")
df = pd.json_normalize(df["fragments"])
df["lines"] = df["lines"].apply(lambda x: x[0])

sound = AudioSegment.from_mp3("data/test/test_0.mp3")

segments = df[["begin", "end"]].to_dict("records")

for segment in segments:
    split = sound[(float(segment["begin"]) * 1000) : (float(segment["end"]) * 1000)]
    split.export(f"data/test/test_{segment['begin']}.mp3", format="mp3")
