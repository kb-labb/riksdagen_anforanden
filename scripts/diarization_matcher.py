import sys
import numpy as np
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import print_overlapping_segments


df = pd.read_parquet("data/df_inference_bleu.parquet")
df2 = pd.read_parquet("data/df_anftext_bleu.parquet")


def create_vars(df):
    df["start_ngram"] = df["contiguous_ngram_match"].apply(lambda x: x[0]).astype("Int32")
    df["start_fuzzy"] = df["contiguous_fuzzy_match"].apply(lambda x: x[0]).astype("Int32")
    df["end_ngram"] = df["contiguous_ngram_match"].apply(lambda x: x[1]).astype("Int32") - 1
    df["end_fuzzy"] = df["contiguous_fuzzy_match"].apply(lambda x: x[1]).astype("Int32") - 1
    df["fuzzy_score"] = df["contiguous_fuzzy_match"].apply(lambda x: x[2])

    df = df.drop(columns=["contiguous_ngram_match", "contiguous_fuzzy_match"])
    return df


df = create_vars(df)
df2 = create_vars(df2)

df = pd.merge(
    df,
    df2[
        [
            "dokid",
            "anforande_nummer",
            "start_ngram",
            "start_fuzzy",
            "end_ngram",
            "end_fuzzy",
            "fuzzy_score",
        ]
    ],
    on=["dokid", "anforande_nummer"],
    suffixes=("_i", "_a"),
    how="left",
)

df["nr_words_i"] = df["anftext_inference"].str.split().str.len()
df["nr_words_a"] = df["anftext_normalized"].str.split().str.len()

df = df[~((df["start_ngram_i"].isna()) & (df["start_fuzzy_i"]).isna())].reset_index(drop=True)


def span_calculator(x, threshold=92, return_column="start", source_text="anftext_inference"):
    """
    Determine a single start or end point for a speech segment based on ngram and fuzzy
    text matching methods.
    """

    if source_text == "anftext_normalized":
        ngram_column = return_column + "_" + "ngram" + "_" + "a"
        fuzzy_column = return_column + "_" + "fuzzy" + "_" + "a"
        fuzzy_score_col = "fuzzy_score" + "_" + "a"
    elif source_text == "anftext_inference":
        ngram_column = return_column + "_" + "ngram" + "_" + "i"
        fuzzy_column = return_column + "_" + "fuzzy" + "_" + "i"
        fuzzy_score_col = "fuzzy_score" + "_" + "i"

    # Choose either max or min value for start or end
    # Max for end, min for start
    if return_column == "end":
        f = max
    elif return_column == "start":
        f = min

    if isinstance(x[fuzzy_column], pd._libs.missing.NAType) or isinstance(
        x[fuzzy_score_col], pd._libs.missing.NAType
    ):
        return x[ngram_column]
    elif isinstance(x[ngram_column], pd._libs.missing.NAType):
        return x[fuzzy_column]
    elif x[fuzzy_score_col] <= threshold:
        return x[ngram_column]
    elif x[fuzzy_score_col] > threshold:
        return f(x[ngram_column], x[fuzzy_column])


df["start_text_i"] = (
    df[["start_ngram_i", "start_fuzzy_i", "fuzzy_score_i"]]
    .apply(
        lambda x: span_calculator(x, return_column="start", source_text="anftext_inference"),
        axis=1,
    )
    .astype("Int32")
)

df["end_text_i"] = (
    df[["end_ngram_i", "end_fuzzy_i", "fuzzy_score_i"]]
    .apply(
        lambda x: span_calculator(x, return_column="end", source_text="anftext_inference"),
        axis=1,
    )
    .astype("Int32")
)

df["start_text_a"] = (
    df[["start_ngram_a", "start_fuzzy_a", "fuzzy_score_a"]]
    .apply(
        lambda x: span_calculator(x, return_column="start", source_text="anftext_normalized"),
        axis=1,
    )
    .astype("Int32")
)

df["end_text_a"] = (
    df[["end_ngram_a", "end_fuzzy_a", "fuzzy_score_a"]]
    .apply(
        lambda x: span_calculator(x, return_column="end", source_text="anftext_normalized"),
        axis=1,
    )
    .astype("Int32")
)

df["start_text_time"] = df[["start_text_i", "chunks"]].apply(
    lambda x: x.chunks[x.start_text_i]["timestamp"][0], axis=1
)
df["end_text_time"] = df[["end_text_i", "chunks"]].apply(
    lambda x: x.chunks[x.end_text_i - 1]["timestamp"][1], axis=1
)
df["start_text_time"] = df["start_text_time"] + df["start"]
df["end_text_time"] = df["end_text_time"] + df["start"]
df["end"] = df["start"] + df["duration"]
df["anftext_coverage_ratio"] = (df["end_text_a"] - df["start_text_a"]) / df["nr_words_a"]
df = df.drop(
    columns=[
        "start_ngram_i",
        "start_fuzzy_i",
        "end_ngram_i",
        "end_fuzzy_i",
        "start_ngram_a",
        "start_fuzzy_a",
        "end_ngram_a",
        "end_fuzzy_a",
    ]
)


"""
Join speaker diarization inference with the text based matching metrics.
"""
df_speakers = pd.read_parquet("data/df_speakers.parquet")

df_speakers["duration"] = df_speakers["end"] - df_speakers["start"]
df_speakers = df_speakers[(df_speakers["duration"] >= 0.8)].reset_index(drop=True)
df_speakers["nr_speakers"] = df_speakers.groupby(["dokid", "anforande_nummer"])["label"].transform(
    "nunique"
)

df_contiguous = []
for group_variables, df_group in tqdm(
    df_speakers.groupby(["dokid", "anforande_nummer"]),
    total=len(df_speakers.groupby(["dokid", "anforande_nummer"])),
):
    # Subset first and last segment of a contiguous speech sequence by the same speaker
    # within a speech audio file duration.
    df_group = df_group.copy()
    df_group = df_group[
        (df_group["label"] != df_group["label"].shift())
        | (df_group["label"] != df_group["label"].shift(-1))
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
df_contiguous["start_segment"] = df_contiguous.groupby(
    ["dokid", "anforande_nummer", "speech_segment_id"]
)["start"].transform("min")
# End of speech segment
df_contiguous["end_segment"] = df_contiguous.groupby(
    ["dokid", "anforande_nummer", "speech_segment_id"]
)["end"].transform("max")

# Merge with speaker diarization inference
df_contiguous = df_contiguous.reset_index(drop=True)
df_c = pd.merge(
    df_contiguous,
    df[
        [
            "dokid",
            "anforande_nummer",
            "start",
            "end",
            "start_text_time",
            "end_text_time",
            "bleu_score",
            "debatedate",
            "anftext_coverage_ratio",
            "nr_words_a",
            "nr_words_i",
        ]
    ],
    on=["dokid", "anforande_nummer"],
    how="left",
    suffixes=("_diarization", "_metadata"),
)

# We filtered out anforanden with bleu_score < 0.1 in text_matcher.py
df_c = df_c[~(df_c["start_text_time"].isna())].reset_index(drop=True)

df_c["start_segment"] = df_c["start_segment"] + df_c["start_metadata"]
df_c["end_segment"] = df_c["end_segment"] + df_c["start_metadata"]

# Non-overlapping segments will get negative len_overlap
df_c["len_overlap"] = df_c[["end_segment", "end_text_time"]].min(axis=1) - df_c[
    ["start_segment", "start_text_time"]
].max(axis=1)

df_c["len_total"] = df_c[["end_segment", "end_text_time"]].max(axis=1) - df_c[
    ["start_segment", "start_text_time"]
].min(axis=1)

df_c["overlap_ratio"] = df_c["len_overlap"] / df_c["len_total"]

# df_c.loc[
#     85000:85050,
#     [
#         "dokid",
#         "anforande_nummer",
#         "label",
#         "speech_segment_id",
#         "start_x",
#         "end_x",
#         "start_y",
#         "end_y",
#         "start_segment",
#         "end_segment",
#         "start_text_time",
#         "end_text_time",
#         "len_overlap",
#         "len_total",
#         "overlap_ratio",
#     ],
# ]

# One obs per unique speech segment
df_segments = (
    df_c.groupby(["dokid", "anforande_nummer", "label", "speech_segment_id"]).first().reset_index()
)

df_segments["duration"] = df_segments["end_segment"] - df_segments["start_segment"]

df_segments["max_segment_id"] = df_segments.groupby(["dokid", "anforande_nummer"])[
    "speech_segment_id"
].transform("max")
df_segments["min_segment_id"] = df_segments.groupby(["dokid", "anforande_nummer"])[
    "speech_segment_id"
].transform("min")

# Our speech of interest is between two other speakers
df_segments["is_between_speeches"] = (
    df_segments["speech_segment_id"] > df_segments["min_segment_id"]
) & (df_segments["speech_segment_id"] < df_segments["max_segment_id"])

# If True, then a different speaker speaks at the end of the audio file
df_segments["exists_speech_after"] = (
    df_segments["speech_segment_id"] < df_segments["max_segment_id"]
)
# If True, then a different speaker speaks at the start of the audio file
df_segments["exists_speech_before"] = (
    df_segments["speech_segment_id"] > df_segments["min_segment_id"]
)
df_segments = df_segments.drop(
    columns=["min_segment_id", "max_segment_id", "start_diarization", "end_diarization"]
)

df_segments["start_margin"] = df_segments["start_segment"] - df_segments["start_metadata"]
df_segments["end_margin"] = df_segments["end_metadata"] - df_segments["end_segment"]
df_segments["speech_duration_ratio"] = df_segments["duration"] / (
    df_segments["end_metadata"] - df_segments["start_metadata"]
)

df_segments
df_train = df_segments.loc[
    (df_segments["anftext_coverage_ratio"] > 0.95)
    & (df_segments["overlap_ratio"] > 0.94)
    & (df_segments["speech_duration_ratio"] > 0.9)
    & (df_segments["duration"] > 20),
    [
        "dokid",
        "anforande_nummer",
        "label",
        "speech_segment_id",
        "start_metadata",
        "end_metadata",
        "start_segment",
        "end_segment",
        "start_text_time",
        "end_text_time",
        "len_overlap",
        "len_total",
        "overlap_ratio",
        "anftext_coverage_ratio",
        "bleu_score",
        "start_margin",
        "end_margin",
        "filename_anforande_audio",
        "duration",
        "nr_words_a",
        "nr_words_i",
    ],
]

df_train

df_train["words_per_sec"] = df_train["nr_words_a"] / df_train["duration"]
df_train["words_per_sec"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
df_train[df_train["end_margin"] < 0.56][50:70]
df_train[df_train["end_margin"] < 0.56].iloc[50:70, 6:]

df_train = pd.merge(
    df_train,
    df[["dokid", "anforande_nummer", "debatedate", "text"]],
    on=["dokid", "anforande_nummer"],
)
# os.makedirs("data/train", exist_ok=True)
df_train.reset_index(drop=True).to_parquet("data/df_train.parquet")

df_segments[df_segments["dokid"] == "GR10190"]
df_c[df_c["dokid"] == "H810792"]

df_segments[
    df_segments["is_between_speeches"]
    | df_segments["exists_speech_after"]
    | df_segments["exists_speech_before"]
]

df_segments[df_segments["exists_speech_after"] & (df_segments["overlap_ratio"] > 0.1)].iloc[
    :, 0:10
]["filename_anforande_audio"][174662]


df_segments[
    (df_segments["anftext_coverage_ratio"] > 0.95)
    & (df_segments["overlap_ratio"] > 0.94)
    & (df_segments["speech_duration_ratio"] > 0.9)
    & (df_segments["duration"] > 20)
    & df_segments["exists_speech_before"]
]

df_segments[
    (df_segments["overlap_ratio"] > 0.9) & (df_segments["anftext_coverage_ratio"] > 0.92)
].iloc[:, 4:16]
df_segments[(df_segments["anftext_coverage_ratio"] > 0.1)]
df_segments.loc[
    (df_segments["overlap_ratio"] > 0.7)
    & (df_segments["end_text_time"] - df_segments["start_text_time"] > 20),
    [
        "dokid",
        "anforande_nummer",
        "label",
        "speech_segment_id",
        "start_x",
        "end_x",
        "start_y",
        "end_y",
        "start_segment",
        "end_segment",
        "start_text_time",
        "end_text_time",
        "len_overlap",
        "len_total",
        "overlap_ratio",
        "bleu_score",
    ],
][97300:97350]
df_c[df_c["dokid"] == "H501JuU21"]


# Subset the duplicate observations in the Dataframe df_c
df_c[df_c.duplicated()]

df_c

df[df.iloc[:, 0:20].duplicated()]

df_c[60000:60050]

df_speakers[df_speakers["dokid"] == "GR10162"][0:20]

df[df["dokid"] == "GV10428"]["debatedate"]


# df["year"] = df["debatedate"].dt.year
# print(df.groupby("year", as_index=False)["bleu_score"].mean().to_markdown(index=False))
# df.groupby("year", as_index=False)["bleu_score"].median()


# text1 = df[df["bleu_score"] > 0.2]["anftext_normalized"].iloc[1]
# text2 = df[df["bleu_score"] > 0.2]["anftext_inference"].iloc[1]

# text2 = df[(df["bleu_score"] < 0.1) & (df["bleu_score"] >= 0.05)]["anftext_inference"].iloc[203]
# text1 = df[(df["bleu_score"] < 0.1) & (df["bleu_score"] >= 0.05)]["anftext_normalized"].iloc[203]


print_overlapping_segments(df, 36, method="ngram", column="anftext_inference")
print_overlapping_segments(df, 36, method="fuzzy", column="anftext_inference")
print_overlapping_segments(df, 36, method="ngram", column="anftext_normalized")
print_overlapping_segments(df, 36, method="fuzzy", column="anftext_normalized")
df["anftext_normalized"][36]


#### Mail to the Swedish parliament ####

# Duplicated rows. Different speeches in same debate sometiems have same anforande_nummer,
# but with different metadata.
df2[df2[["dokid", "anforande_nummer"]].duplicated(keep=False)].iloc[0:50, 0:12]
