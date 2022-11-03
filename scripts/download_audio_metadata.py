import multiprocessing as mp
import pandas as pd
import locale
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import get_audio_metadata
from src.data import coalesce_columns
from tqdm.contrib.concurrent import process_map  # multiprocessing from tqdm

locale.setlocale(locale.LC_ALL, "sv_SE.UTF-8")  # Swedish date format

df = pd.read_parquet("data/df_anforanden_metadata.parquet")
df = df[~pd.isna(df["rel_dok_id"])].reset_index(drop=True)

# Some anforanden have multiple rel_dok_ids, we select the first one
first_rel_dok_id = df[df["rel_dok_id"].str.contains(",")]["rel_dok_id"].str.extract("(.*?)(?=, )")
df.loc[df["rel_dok_id"].str.contains(","), "rel_dok_id"] = first_rel_dok_id.iloc[:, 0].tolist()

# Downlaod audio metadata from unique rel_dok_ids (debates)
df_list = process_map(
    get_audio_metadata,
    df["rel_dok_id"].unique().tolist(),
    max_workers=mp.cpu_count(),
    chunksize=20,
)

df_audiometa = pd.concat(df_list, axis=0)
df_audiometa = df_audiometa.reset_index(drop=True)
df_audiometa["debatedate"] = pd.to_datetime(df_audiometa["debatedate"], format="%d %B %Y")
df_audiometa.loc[df_audiometa["anftext"] == "", "anftext"] = None

# # Add direct timestamped link to webb-tv to start video where a speech begins
# df_audiometa["debateurl_timestamp"] = (
#     "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
#     + df_audiometa["dokid"]
#     + "&start="
#     + df_audiometa["start"].astype(str)
#     + "&end="
#     + (df_audiometa["start"] + df_audiometa["duration"]).astype(str)
# )


# Some speech texts are missing from audio metadata, we add them from df_anforanden_metadata
df_audiometa = df_audiometa.rename(columns={"number": "anforande_nummer"})

df_audiometa = df_audiometa.merge(
    df[["rel_dok_id", "anforande_nummer", "anforandetext"]],
    left_on=["rel_dok_id", "anforande_nummer"],
    right_on=["rel_dok_id", "anforande_nummer"],
    how="left",
)

# Replace NaN in anftext column with text from anforandetext
df_audiometa = coalesce_columns(df_audiometa, col1="anftext", col2="anforandetext")


df_audiometa.to_parquet("data/df_audio_metadata.parquet", index=False)
