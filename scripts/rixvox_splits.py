import os
from pathlib import Path
import re
import multiprocessing as mp
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

df = pd.read_parquet("data/df_final_riksvox.parquet")
df["filename_anforande_json"] = df["filename_anforande_audio"].str.extract(r"(.+?)\.") + ".json"
# df = df[78000:].reset_index(drop=True)


def read_force_alignments(filename_json):
    df_line = pd.read_json("data/audio/" + filename_json)
    df_line = pd.json_normalize(df_line["fragments"])
    df_line["filename_anforande_json"] = filename_json
    df_line["lines"] = df_line["lines"].str[0]  # Remove list brackets.
    return df_line


force_alignment_jsons = df["filename_anforande_json"]

with mp.Pool(26) as p:
    df_lines = p.map(
        read_force_alignments,
        tqdm(force_alignment_jsons, total=len(force_alignment_jsons)),
        chunksize=4,
    )

df_lines = pd.concat(df_lines)
df_lines = pd.merge(
    df_lines,
    df[
        [
            "dokid",
            "anforande_nummer",
            "speaker",
            "party",
            "sex",
            "electoral_district",
            "birth_year",
            "intressent_id",
            "filename_anforande_audio",
            "filename_anforande_json",
            "speaker_from_id",
            "speaker_audio_meta",
            "debatedate",
            "bleu_score",
        ]
    ],
    on="filename_anforande_json",
    how="left",
)

df_lines = df_lines.drop(columns="children")
df_lines["begin"] = df_lines["begin"].astype(float)
df_lines["end"] = df_lines["end"].astype(float)
df_lines["duration"] = df_lines["end"] - df_lines["begin"]
df_lines = (
    df_lines.groupby("filename_anforande_audio", as_index=False).apply(lambda x: x.iloc[1:]).reset_index(drop=True)
)

df_groups = []
for group, df_group in tqdm(
    df_lines.groupby("filename_anforande_audio"),
    total=df_lines.groupby("filename_anforande_audio").ngroups,
):
    observation_nr = 0
    lastvalue = 0
    newcum = []
    observation_nrs = []
    for duration in df_group["duration"]:
        if lastvalue + duration >= 30:
            lastvalue = duration
            observation_nr += 1
        else:
            lastvalue += duration
        newcum.append(lastvalue)
        observation_nrs.append(observation_nr)

    df_group["observation_nr"] = observation_nrs
    df_group["cumsum"] = newcum
    df_groups.append(df_group)

df_groups = pd.concat(df_groups)
df_groups = df_groups[df_groups["duration"] < 30].reset_index(drop=True)

df_obs = (
    df_groups.groupby(["filename_anforande_audio", "observation_nr"], group_keys=False)["lines"]
    .apply(lambda x: list(x))
    .reset_index()
)

df_obs["start"] = df_groups.groupby(["filename_anforande_audio", "observation_nr"]).first().reset_index()["begin"]
df_obs["end"] = df_groups.groupby(["filename_anforande_audio", "observation_nr"]).last().reset_index()["end"]
df_obs["duration"] = df_groups.groupby(["filename_anforande_audio", "observation_nr"]).last().reset_index()["cumsum"]
df_obs[
    [
        "dokid",
        "anforande_nummer",
        "debatedate",
        "speaker",
        "party",
        "sex",
        "electoral_district",
        "birth_year",
        "intressent_id",
        "speaker_from_id",
        "speaker_audio_meta",
        "bleu_score",
    ]
] = (
    df_groups.groupby(["filename_anforande_audio", "observation_nr"])
    .first()
    .reset_index()[
        [
            "dokid",
            "anforande_nummer",
            "debatedate",
            "speaker",
            "party",
            "sex",
            "electoral_district",
            "birth_year",
            "intressent_id",
            "speaker_from_id",
            "speaker_audio_meta",
            "bleu_score",
        ]
    ]
)
df_obs["lines"] = df_obs["lines"].str.join(" ")


def split_audio_by_line(df, audio_dir="data/rixvox", file_exists_check=False):
    """
    Split audio file by anförande (speech) and save to disk in folder for specific dokid.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid
            and anforande_nummer. df["filename_anforande_audio"] looks like
            "GR01SFU15/2442210140028274621_aud_1398_1901.wav", i.e. {dokid}/{filename}.
        audio_dir (str): Path to directory where audio files should be saved.
        file_exists_check (bool): If True, checks whether split file already exists and
            skips it. When False, reprocesses all files.
    """
    # Match until first _ but don't include _.
    filename_anf = df["filename_anforande_audio"].iloc[0]
    anf_nr = df["anforande_nummer"].iloc[0]
    segments = df[["start", "end", "anforande_nummer"]].to_dict(orient="records")
    sound = AudioSegment.from_wav(os.path.join("data/audio", filename_anf))
    # Match until first _ but don't include _.
    # GR01SFU15/2442210140028274621_aud_1398_1901.wav -> GR01SFU15/2442210140028274621
    filename_base = re.match(r"(.+?)(?=_)", filename_anf).group(0)

    filenames_speeches = []
    for segment in segments:
        start = float(segment["start"]) * 1000  # ms
        end = float(segment["end"]) * 1000
        split = sound[start:end]

        filename = Path(filename_base).parent / Path(filename_base).stem  # Filename without extension.

        filename_speech = Path(f"{filename}_anf{anf_nr}_{int(round(start/1000))}_{int(round(end/1000))}.wav")

        if file_exists_check:
            if os.path.exists(os.path.join(audio_dir, filename_speech)):
                print(f"File {filename_speech} already exists.")
                continue

        os.makedirs(os.path.join(audio_dir, filename.parent), exist_ok=True)
        filenames_speeches.append(filename_speech)
        split.export(os.path.join(audio_dir, filename_speech), format="wav", bitrate="16k")

    df["filename_train_audio"] = filenames_speeches
    df["filename_train_audio"] = df["filename_train_audio"].astype(str)
    print(f"{filename_speech.parent} complete", end="\r", flush=True)
    return df


df_train = df_obs.groupby(["dokid", "anforande_nummer"])
df_list = [df_train.get_group(x) for x in df_train.groups]  # list of dfs, one for each dokid


with mp.Pool(28) as pool:
    df_list = pool.map(
        split_audio_by_line,
        tqdm(
            df_list,
            total=df_obs.groupby(["dokid", "anforande_nummer"]).ngroups,
        ),
        chunksize=1,
    )

df_list = pd.concat(df_list).reset_index(drop=True)

df_list = df_list.rename(columns={"filename_train_audio": "filename", "lines": "text"})
df_list[
    [
        "dokid",
        "anforande_nummer",
        "observation_nr",
        "speaker",
        "party",
        "sex",
        "debatedate",
        "electoral_district",
        "birth_year",
        "intressent_id",
        "speaker_from_id",
        "speaker_audio_meta",
        "text",
        "start",
        "end",
        "duration",
        "bleu_score",
        "filename",
    ]
].reset_index(drop=True).to_parquet("data/df_train.parquet", index=False)

# df_obs["filename_train_audio"] = df_obs["filename_anforande_audio"].str.extract("(.+?)(?=_)")
# df_obs["filename_train_audio"] = (
#     df_obs["filename_train_audio"]
#     + "_anf"
#     + df_obs["anforande_nummer"].astype(str)
#     + "_"
#     + round(df_obs["start"]).astype(int).astype(str)
#     + "_"
#     + round(df_obs["end"]).astype(int).astype(str)
#     + ".wav"
# )
