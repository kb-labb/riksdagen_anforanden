import shutil
import os
import pandas as pd
from tqdm import tqdm


def preprocess_audio_metadata(speech_metadata):
    """
    Preprocess the speech_metadata dict to a pandas dataframe.

    Args:
        speech_metadata (dict): Nested metadata fields with transcribed texts, media file
            URLs and more.

    Returns:
        pd.DataFrame: A pandas dataframe with the relevant metadata fields.
    """

    df = pd.DataFrame(speech_metadata["videodata"])
    df = df.explode("speakers").reset_index(drop=True)
    df_files = pd.json_normalize(df["streams"], ["files"])
    df_speakers = pd.json_normalize(df["speakers"])
    df = df.drop(columns=["streams", "speakers"])
    df = pd.concat([df, df_files], axis=1)
    df = pd.concat([df, df_speakers], axis=1)

    df = df[
        [
            "dokid",
            "party",
            "start",
            "duration",
            "debateseconds",
            "title",
            "text",
            "debatename",
            "debatedate",
            "url",
            "debateurl",
            "id",
            "subid",
            "audiofileurl",
            "downloadfileurl",
            "debatetype",
            "number",
            "anftext",
        ]
    ]

    df["anftext"] = df["anftext"].str.replace(
        r"STYLEREF Kantrubrik \\+\* MERGEFORMAT", "", regex=True
    )
    df["anftext"] = df["anftext"].str.replace(r"Svar på interpellationer", "", regex=True)
    df["anftext"] = df["anftext"].str.replace(r"<.*?>", " ", regex=True)  # Remove HTML tags
    # Remove text within parentheses
    df["anftext"] = df["anftext"].str.replace(r"\(.*?\)", "", regex=True)
    df["anftext"] = df["anftext"].str.strip()
    # Remove multiple spaces
    df["anftext"] = df["anftext"].str.replace(r"(\s){2,}", " ", regex=True)

    return df


def audio_to_dokid_folder(df, folder="data/audio"):
    """
    Move audio files to a folder named after the dokid.

    Args:
        df (pd.DataFrame): A pandas dataframe with the relevant metadata fields.
    """

    # Only keep the first row for each dokid, they all have same audio file
    df = df.groupby("dokid").first().reset_index()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dokid = row["dokid"]
        audiofileurl = row["audiofileurl"]
        filename = audiofileurl.rsplit("/", 1)[1]
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, dokid, filename)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        else:
            if os.path.exists(dst):
                print(f"File already in destination folder: {dst}")
            else:
                print(f"File not found: {src}")
