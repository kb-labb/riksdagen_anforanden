import shutil
import os
import pandas as pd
import string
import re
from tqdm import tqdm
from num2words import num2words


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

    df = preprocess_text(df, is_audio_metadata=True)

    return df


def preprocess_text(df, textcol="anftext", is_audio_metadata=False):
    """
    Preprocess the text field.

    Args:
        df (pd.DataFrame): A pandas dataframe that contains text column with speeches.
        textcol (str): The name of the text column.

    Returns:
        pd.DataFrame: A pandas dataframe with preprocessed text column.
    """

    # Remove all text within <p> tags that contain "STYLEREF".
    # These are headers mistakenly included in the text as paragraphs.
    df[textcol] = df[textcol].str.replace(r"(<p> STYLEREF.*?</p>)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(<p>Gransknings- STYLEREF.*?</p>)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(<p><em></em><em> STYLEREF.*?</p>)", "", regex=True)

    # Some extra headers that don't contain "STYLEREF", but are still in <p> tags.
    # We grab the headers from the header column and remove "<p>{header}</p>" from the text column.
    # data/headers.csv is created in scripts/preprocess_speeches_metadata.py.
    headers = pd.read_csv("data/headers.csv")["avsnittsrubrik"].tolist()

    for header in headers:
        remove_header_p = f"<p>{header}</p>"
        df[textcol] = df[textcol].str.replace(remove_header_p, "", regex=False)

    df[textcol] = df[textcol].str.replace(r"<.*?>", " ", regex=True)  # Remove HTML tags
    # Remove text within parentheses, e.g. (applåder)
    df[textcol] = df[textcol].str.replace(r"\(.*?\)", "", regex=True)

    # Speaker of the house or other text not part of actual speech.
    # Found at the end of a transcript.
    df[textcol] = df[textcol].str.replace(r"(Interpellationsdebatten var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Partiledardebatten var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Frågestunden var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Överläggningen var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Den särskilda debatten var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Statsministerns frågestund var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Återrapporteringen var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Den muntliga frågestunden var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Den utrikespolitiska debatten var [h|d]ärmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Den allmänpolitiska debatten var härmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Den aktuella debatten var härmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(Informationen var härmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(
        r"(Den EU-politiska (partiledar)?debatten var härmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Debatten med anledning av (vår|budget)propositionens avlämnande var härmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(r"(I detta anförande instämde.*)", "", regex=True)

    df[textcol] = df[textcol].str.strip()

    # Normalize text
    df[textcol] = df[textcol].str.normalize("NFKC")  # Normalize unicode characters
    # Remove multiple spaces
    df[textcol] = df[textcol].str.replace(r"(\s){2,}", " ", regex=True)
    # Replace &amp; with &
    df[textcol] = df[textcol].str.replace(r"&amp;", "&", regex=True)

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


def audiofile_exists(dokid, filename, folder="data/audio"):
    """
    Check if audio file exists.
    Can exist in either {folder}/{dokid}/{filename} or {folder}/{filename}.

    Args:
        dokid (str): The dokid of the speech. dokid is the folder where the
            audiofile is moved after being download.
        filename (str): The filename of the audio file.
        folder (str): The folder where the audio files are stored.

    Returns:
        bool: True if audio file exists, otherwise False.
    """

    src = os.path.join(folder, filename)
    dst = os.path.join(folder, dokid, filename)

    if os.path.exists(src) or os.path.exists(dst):
        return True
    else:
        return False


def coalesce_columns(df, col1="anftext", col2="anforandetext"):
    """
    Coalesce text columns in df, replacing NaN values (i.e. missing speeches)
    in first column with values from second column.

    Args:
        df (pd.DataFrame): A pandas dataframe with the relevant metadata fields.
        col1 (str): The name of the 1st text column in df whose NaNs we are filling.
        col2 (str): The name of the 2nd text column in df.

    Returns:
        pd.DataFrame: A pandas dataframe with the coalesced text column.
    """

    df[col1] = df[col1].fillna(df[col2])
    df = df.drop(columns=[col2])

    return df


def normalize_text(df, column_in, column_out):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text.

    Args:
        df (pd.DataFrame): A pandas dataframe that contains text column anftext with speeches.
        column_in (str): The name of the text column to normalize.
        column_out (str): The name of the normalized text column.
    Returns:
        pd.DataFrame: A pandas dataframe with normalized text column `column_out`.
    """
    df[column_out] = df[column_in].apply(
        lambda x: None if x is None else x.translate(str.maketrans("", "", string.punctuation))
    )
    df[column_out] = df[column_out].str.lower()
    # df[column_out] = df[column_out].str.replace("\xa0", " ")
    df[column_out] = df[column_out].str.normalize("NFKC")  # Normalize unicode characters
    # Remove hyphen between words
    df[column_out] = df[column_out].str.replace("(?<=\w)-(?=\w)", " ", regex=True)
    # Remove multiple spaces and replace with single space
    df[column_out] = df[column_out].str.replace(" +", " ", regex=True)
    # Remove whitespace between numbers
    df[column_out] = df[column_out].str.replace("(?<=\d) (?=\d)", "", regex=True)
    # Convert numbers to words
    df[column_out] = df[column_out].apply(
        lambda x: None if x is None else re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), x)
    )

    return df
