import os
import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser(
    description="""Read json files of riksdagens anföranden, save relevant metadata fields to file."""
)
parser.add_argument("-f", "--folder", type=str, default="data/json")
parser.add_argument("-d", "--dest_folder", type=str, default="data")
args = parser.parse_args()

json_files = [
    os.path.join(args.folder, filename)
    for filename in os.listdir(args.folder)
    if filename.endswith(".json")
]

json_anforanden = []
for file in json_files:
    with open(file) as f:
        json_anforanden.append(json.load(f)["anforande"])

df = pd.json_normalize(json_anforanden)
df = df.drop(columns=["anforandetext"])

df = df.sort_values(["dok_id", "anforande_nummer"]).reset_index(drop=True)
df.loc[df["rel_dok_id"] == "", "rel_dok_id"] = None

df.to_parquet(os.path.join(args.dest_folder, "df_anforanden_metadata.parquet"))
