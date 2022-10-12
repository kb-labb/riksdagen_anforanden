import pandas as pd


def preprocess_audio_metadata(anforande_metadata):
    """
    Preprocess the anforande_metadata dict to a pandas dataframe.

    Args:
        anforande_metadata (dict): Nested metadata fields with transcribed texts, media file
        URLs and more.

    Returns:
        pd.DataFrame: A pandas dataframe with the relevant metadata fields.
    """

    df = pd.DataFrame(anforande_metadata["videodata"])
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
