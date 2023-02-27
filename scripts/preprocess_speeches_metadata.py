import pandas as pd
import argparse
import json
import os
from tqdm import tqdm
from src.data import preprocess_text

parser = argparse.ArgumentParser(
    description="""Read json files of riksdagens anföranden, save relevant metadata fields to file."""
)
parser.add_argument("-f", "--folder", type=str, default="data/json")
parser.add_argument("-d", "--dest_folder", type=str, default="data")
args = parser.parse_args()

json_files = [
    os.path.join(args.folder, filename) for filename in os.listdir(args.folder) if filename.endswith(".json")
]

json_speeches = []

for file in tqdm(json_files):
    with open(os.path.join(file), "r", encoding="utf-8-sig") as f:
        json_speeches.append(json.load(f)["anforande"])

print("Normalizing json to dataframe...")
df = pd.json_normalize(json_speeches)
# df = df.drop(columns=["anforandetext"])
df["anforande_nummer"] = df["anforande_nummer"].astype(int)

# Headers to clean up when next script is run (download_audio_metadata.py)
headers = df.groupby("avsnittsrubrik").size().sort_values(ascending=False).head(1000)
headers.reset_index().rename(columns={0: "count"}).to_csv("data/headers.csv", index=False)

print("Preprocessing text...")
df = preprocess_text(df, textcol="anforandetext")

df = df.sort_values(["dok_id", "anforande_nummer"]).reset_index(drop=True)
df.loc[df["rel_dok_id"] == "", "rel_dok_id"] = None

print(f"Saving file to {os.path.join(args.dest_folder, 'df_anforanden_metadata.parquet')}")
df.to_parquet(os.path.join(args.dest_folder, "df_anforanden_metadata.parquet"))
