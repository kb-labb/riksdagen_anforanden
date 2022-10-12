import multiprocessing as mp
import pandas as pd
from src.api import get_audio_metadata
from tqdm.contrib.concurrent import process_map  # multiprocessing from tqdm

df = pd.read_parquet("data/df_anforanden_metadata.parquet")
df = df[~pd.isna(df["rel_dok_id"])]

# Some anforanden have multiple rel_dok_ids, we select the first one
first_rel_dok_id = df[df["rel_dok_id"].str.contains(",")]["rel_dok_id"].str.extract("(.*?)(?=, )")
df.loc[df["rel_dok_id"].str.contains(","), "rel_dok_id"] = first_rel_dok_id.iloc[:, 0].tolist()

# Change to unique rel_dok_ids
df_list = process_map(
    get_audio_metadata,
    df["rel_dok_id"].unique().tolist(),
    max_workers=mp.cpu_count(),
    chunksize=20,
)

df_audiometa = pd.concat(df_list, axis=0)
df_audiometa = df_audiometa.reset_index(drop=True)


## Add direct timestamped link to webb-tv to start video where a speech begins
# df_audiometa["debateurl_timestamp"] = (
#     "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
#     + df_audiometa["dokid"]
#     + "&start="
#     + df_audiometa["start"].astype(str)
#     + "&end="
#     + (df_audiometa["start"] + df_audiometa["duration"]).astype(str)
# )

df_audiometa.to_parquet("data/df_audio_metadata.parquet", index=False)
