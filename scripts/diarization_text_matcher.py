import pandas as pd
from tqdm import tqdm

df = pd.read_parquet("data/df_audio_metadata_new.parquet")
df_timestamp = pd.read_parquet("data/df_timestamp_new.parquet")
df_diarization = pd.read_parquet("data/df_speakers_debate.parquet")

# Left join the columns dokid, anforance_nummer, debatedate from df in to df_timestamp
df_timestamp = df_timestamp.merge(
    df[["dokid", "anforande_nummer", "debatedate", "start", "duration"]],
    on=["dokid", "anforande_nummer", "start", "duration"],
    how="left",
)

# Left join df_timestamp in to df_diarization
df_diarization["duration"] = df_diarization["end"] - df_diarization["start"]
df_diarization = df_diarization[df_diarization["duration"] >= 0.6].reset_index(drop=True)

df_contiguous = []
for group_variables, df_group in tqdm(
    df_diarization.groupby(["dokid"]),
    total=len(df_diarization.groupby(["dokid"])),
):
    # Subset first and last segment of a contiguous speech sequence by the same speaker
    # within a speech audio file duration.
    df_group = df_group.copy()
    df_group = df_group[
        (df_group["label"] != df_group["label"].shift()) | (df_group["label"] != df_group["label"].shift(-1))
    ]

    # Unique speech segment ids for each anforande. E.g. if speaker_0 starts, followed by
    # speaker_1, followed by speaker_0 again, then the speech segment ids are 0, 1, 2.
    df_group["speech_segment_id"] = (
        df_group["label"].eq(0) | (df_group["label"] != df_group["label"].shift())
    ).cumsum()

    df_contiguous.append(df_group)


df_contiguous = pd.concat(df_contiguous)

df_contiguous["nr_speech_segments"] = df_contiguous.groupby(["dokid", "anforande_nummer"])[
    "speech_segment_id"
].transform("max")
# Start of contiguous speech segment
df_contiguous["start_segment"] = df_contiguous.groupby(["dokid", "anforande_nummer", "speech_segment_id"])[
    "start"
].transform("min")
# End of speech segment
df_contiguous["end_segment"] = df_contiguous.groupby(["dokid", "anforande_nummer", "speech_segment_id"])[
    "end"
].transform("max")

df_segments = df_contiguous.groupby(["dokid", "label", "speech_segment_id"]).first().reset_index()
df_segments = df_segments.sort_values(["dokid", "speech_segment_id"]).reset_index(drop=True)
df_segments = df_segments.drop(columns=["start", "end", "duration"])
df_segments["duration_segment"] = df_segments["end_segment"] - df_segments["start_segment"]


df_overlap = []
for dokid, df_timestamp_group in tqdm(df_timestamp.groupby("dokid"), total=len(df_timestamp.groupby("dokid"))):
    df_segment_group = df_segments[df_segments["dokid"] == dokid]
    for i, row in df_timestamp_group.iterrows():
        df_segment_group = df_segment_group.copy()
        df_segment_group["start"] = row["start"]
        df_segment_group["duration"] = row["duration"]
        df_segment_group["end"] = row["end"]
        df_segment_group["start_text_time"] = row["start_text_time"]
        df_segment_group["end_text_time"] = row["end_text_time"]
        df_segment_group["anforande_nummer"] = row["anforande_nummer"]

        df_segment_group["duration_text"] = df_segment_group["end_text_time"] - df_segment_group["start_text_time"]
        df_segment_group["duration_overlap"] = df_segment_group[["end_segment", "end_text_time"]].min(
            axis=1
        ) - df_segment_group[["start_segment", "start_text_time"]].max(axis=1)

        df_segment_group["overlap_ratio"] = df_segment_group["duration_overlap"] / df_segment_group["duration_text"]

        overlapping_segments = df_segment_group[df_segment_group["overlap_ratio"] > 0.05]
        df_overlap.append(overlapping_segments)


df_overlap = pd.concat(df_overlap, ignore_index=True)
df_overlap = df_overlap.sort_values(["dokid", "anforande_nummer", "speech_segment_id"]).reset_index(drop=True)

df_list = []
for group_variables, df_group in tqdm(df_overlap.groupby(["dokid", "anforande_nummer"])):
    # Subset first and last segment of a contiguous speech sequence by the same speaker
    # within each dokid and anforande_nummer.
    df_group = df_group.copy()
    df_group = df_group[
        (df_group["label"] != df_group["label"].shift()) | (df_group["label"] != df_group["label"].shift(-1))
    ]
    df_list.append(df_group)

df_overlap = pd.concat(df_list)

# Start of contiguous speech segment
df_overlap["start_segment"] = df_overlap.groupby(["dokid", "anforande_nummer", "label"])["start_segment"].transform(
    "min"
)
# End of speech segment
df_overlap["end_segment"] = df_overlap.groupby(["dokid", "anforande_nummer", "label"])["end_segment"].transform("max")
df_overlap = df_overlap.groupby(["dokid", "label", "anforande_nummer"]).first().reset_index()
df_overlap["duration_segment"] = df_overlap["end_segment"] - df_overlap["start_segment"]


# Count number of anforande_nummer per dokid using .transform("count")
df_overlap["nr_speech_segments"] = df_overlap.groupby(["dokid", "anforande_nummer"])["label"].transform("count")
df_overlap["duration_overlap"] = df_overlap[["end_segment", "end_text_time"]].min(axis=1) - df_overlap[
    ["start_segment", "start_text_time"]
].max(axis=1)
df_overlap["duration_text"] = df_overlap["end_text_time"] - df_overlap["start_text_time"]
df_overlap["overlap_ratio"] = df_overlap["duration_overlap"] / df_overlap["duration_text"]
df_overlap["length_ratio"] = df_overlap["duration_segment"] / df_overlap["duration_text"]

# Group by dokid and anforande_nummer and keep only the row with the highest overlap_ratio
df_overlap = (
    df_overlap.groupby(["dokid", "anforande_nummer"])
    .apply(lambda x: x.loc[x["overlap_ratio"].idxmax()])
    .reset_index(drop=True)
)

# df_overlap.to_parquet("data/df_diarization.parquet", index=False)

df_overlap = pd.read_parquet("data/df_diarization.parquet")

df_overlap["debateurl_timestamp"] = (
    "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
    + df_overlap["dokid"]
    + "&start="
    + df_overlap["start_segment"].astype(str)
    + "&end="
    + (df_overlap["start_segment"] + df_overlap["duration_segment"]).astype(str)
)


# # Left join df_overlap in to df
# df = df.merge(df_overlap, on=["dokid", "anforande_nummer"], how="left")

# df["end"] = df["start"] + df["duration"]
# df["start_diff"] = df["start_segment"] - df["start"]
# df["end_diff"] = df["end_segment"] - df["end"]

# df["debateurl_timestamp"] = (
#     "https://www.riksdagen.se/views/pages/embedpage.aspx?did="
#     + df["dokid"]
#     + "&start="
#     + df["start_segment"].astype(str)
#     + "&end="
#     + (df["start_segment"] + df["duration_segment"]).astype(str)
# )

# pd.set_option("max_colwidth", 120)

# df["debateurl"] = "https://riksdagen.se" + df["debateurl"]

df_overlap[df_overlap["duration_segment"] >= 30]["duration_segment"].sum() / 3600

df_overlap["duration_segment"].sum() / 3600

df_overlap[df_overlap["length_ratio"] > 2]

df = pd.read_parquet("data/df_audio_metadata_new.parquet")
df_overlap = pd.read_parquet("data/df_diarization.parquet")

# Version 1 join
a = df.merge(
    df_overlap.drop(["filename", "start", "duration"], axis=1),
    on=["dokid", "anforande_nummer"],
    how="left",
)

# a.to_parquet("data/df_audio_metadata_final.parquet", index=False)

# Set pd.option to display more of the column
pd.set_option("max_colwidth", 120)

a[
    [
        "dokid",
        "anforande_nummer",
        "start_segment",
        "end_segment",
        "start_text_time",
        "end_text_time",
        "start",
        "duration",
        "duration_overlap",
        "length_ratio",
        "overlap_ratio",
        "text",
        "debateurl_timestamp",
    ]
][14800:14850]


a[(a[["dokid", "anforande_nummer"]].duplicated(keep=False)) & (~a["start_segment"].isna())].sort_values(
    ["dokid", "anforande_nummer"]
).iloc[50:60][
    [
        "dokid",
        "anforande_nummer",
        "start_segment",
        "start_text_time",
        "end_segment",
        "end_text_time",
        "start",
        "duration",
        "duration_overlap",
        "length_ratio",
        "overlap_ratio",
        "text",
        "debateurl_timestamp",
    ]
].values

a[(a[["dokid", "anforande_nummer"]].duplicated())].sort_values(["dokid", "anforande_nummer"]).sort_values(
    ["dokid", "anforande_nummer"]
).iloc[0:50]


a["start_segment"].isna().sum()

# Version 2 join
b = df.merge(
    df_overlap.drop("filename", axis=1),
    on=["dokid", "anforande_nummer", "start", "duration"],
    how="left",
)

b[b[["dokid", "anforande_nummer"]].duplicated()][0:50]

b["start_segment"].isna().sum()
b[(b["dokid"] == "GU10162") & (b["anforande_nummer"] == 21)]

a.columns
a[a["start_text_time"].isna()]

df_overlap[
    (df_overlap["length_ratio"] < 1.5)
    & (df_overlap["length_ratio"] > 0.7)
    & (df_overlap["overlap_ratio"] > 0.7)
    & (df_overlap["duration_segment"] > 30)
]["duration_segment"].sum() / 3600


df_overlap.columns
