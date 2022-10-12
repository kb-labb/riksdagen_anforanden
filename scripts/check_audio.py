import os
import pandas as pd
from pydub import AudioSegment

df = pd.read_parquet("data/df_audio_metadata.parquet")

# df[df["audiofileurl"].str.contains("2442209030025807721_aud.mp3")]["debateurl"].iloc[0]
# df[df["audiofileurl"].str.contains("2442209030025807721_aud.mp3")]
# # Beatrice Ask i ljudfilen.
# df[df["anftext"].str.contains("genomdriva den största satsningen")]


# df[df["audiofileurl"].str.contains("2442210070028068221_aud.mp3")]
# df[df["audiofileurl"].str.contains("2442210070028068221_aud.mp3")]["debateurl"].iloc[0]
# df[df["anftext"].str.contains("realistiskt att helt ta bort fastighetsskatten")].iloc[:, 5:10]


df_sub = df.groupby(["dokid", "audiofileurl"]).first().reset_index()
files = os.listdir("data/audio")

df_sub.columns

df_sub["filename"] = df_sub["audiofileurl"].apply(lambda x: x.rsplit("/")[-1])
df_sub[df_sub["filename"].isin(files)]

sound = AudioSegment.from_mp3("data/audio/2442209030025807721_aud.mp3")

len(sound) / 1000 / 60
