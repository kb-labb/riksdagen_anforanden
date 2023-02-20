import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import coalesce_columns

df = pd.read_parquet("data/df_audio_metadata_new.parquet")
df_person = pd.read_csv("person.csv")

df_person["From"] = pd.to_datetime(df_person["From"])
df_person["Tom"] = pd.to_datetime(df_person["Tom"])

df_person = df_person.groupby("Id").first().reset_index()

df = df.merge(
    df_person[["Id", "Förnamn", "Efternamn", "Parti", "Kön", "Född", "Valkrets"]],
    left_on="intressent_id",
    right_on="Id",
    how="left",
)

df["speaker"] = df["Förnamn"] + " " + df["Efternamn"]

# Remove titles and party abbrevations to make it as similar as possible in format to df["speaker"]
# Temporary column to be able to join names from "text" column to "speaker" column in a common format.
df["speaker_audio_meta"] = df["text"]
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace(".+?minister", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace(".+?min\.", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace(".+?rådet", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace(".+?[T|t]alman", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace("Talman", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.replace(r"\(.*\)", "")
df["speaker_audio_meta"] = df["speaker_audio_meta"].str.strip()


df["speaker_from_id"] = True
df.loc[df["speaker"].isna(), "speaker_from_id"] = False
df = coalesce_columns(df, col1="speaker", col2="speaker_audio_meta")
df = coalesce_columns(df, col1="Parti", col2="party")
df["speaker_audio_meta"] = df["text"]
df = df.drop(columns="text")

df["Född"] = df["Född"].astype("Int64")
df["Parti"] = df["Parti"].str.upper()
df.loc[df["Kön"] == "man", "sex"] = "male"
df.loc[df["Kön"] == "kvinna", "sex"] = "female"

df = df.rename(columns={"Valkrets": "electoral_district", "Född": "birth_year", "Parti": "party"})
df = df.drop(columns=["Förnamn", "Efternamn", "Kön", "Id"])


df[["speaker", "intressent_id", "sex", "electoral_district", "birth_year"]] = (
    df[["speaker", "intressent_id", "sex", "electoral_district", "birth_year"]]
    .groupby("speaker", group_keys=False)
    .apply(lambda x: x.fillna(x.mode().iloc[0]))
    .reset_index(drop=True)
)

df.to_parquet("df_audio_metadata_new.parquet", index=False)
