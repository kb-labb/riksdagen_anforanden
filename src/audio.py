import os
from pathlib import Path
from pydub import AudioSegment
from nltk import sent_tokenize


def split_audio_by_speech(df, audio_dir="data/audio"):
    """
    Split audio file by anförande (speech) and save to disk in folder for specific dokid.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            df["filename"] looks like "H901KrU5/2442204200009516121_aud.mp3",
            i.e. {dokid}/{filename}.
        audio_dir (str): Path to directory where audio files should be saved.
    """

    filename_dokid = df["filename"].iloc[0]
    segments = df[["start", "duration"]].to_dict(orient="records")
    sound = AudioSegment.from_mp3(os.path.join(audio_dir, filename_dokid))

    filenames_speeches = []
    for segment in segments:
        start = float(segment["start"]) * 1000  # ms
        end = (float(segment["start"]) + float(segment["duration"])) * 1000
        split = sound[start:end]

        filename = (
            Path(filename_dokid).parent / Path(filename_dokid).stem
        )  # Filename without extension.

        filename_speech = (
            f"{filename}_{segment['start']}_{segment['start'] + segment['duration']}.mp3"
        )

        # if os.path.exists(os.path.join(audio_dir, filename_speech)):
        #     print(f"File {filename_speech} already exists.")
        #     continue

        filenames_speeches.append(filename_speech)
        split.export(os.path.join(audio_dir, filename_speech), format="mp3")

    df["filename_anforande_audio"] = filenames_speeches
    print(filename)
    return df


def get_corrupt_audio_files(df, audio_dir="data/audio"):
    """
    Get list of corrupt audio files that were not able to be force aligned.
    We retry those mp3 files that have no corresponding json sync file.

    Parameters:
        df (pandas.DataFrame): DataFrame with all audio metadata, including
        filenames of audio files.
        audio_dir (str): Path to directory where audio files were saved.
    """

    def json_exists(filename):
        return os.path.exists(
            Path(audio_dir) / Path(filename).parent / f"{Path(filename).stem}.json"
        )

    df["corrupt"] = df["filename_anforande_audio"].apply(lambda x: not json_exists(x))
    return df[df["corrupt"] == True].reset_index(drop=True)


def split_text_by_speech(df, text_dir="data/audio"):
    """
    Split text file by anforande (speech) and save to disk in folder for specific dokid.
    If audio file for speech is saved as "H901KrU5/2442204200009516121_aud_0_233.mp3",
    then text file should be saved as "H901KrU5/2442204200009516121_aud_0_233.txt".

    Assumes split_audio_by_speech() has been run.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            df["anftext"] contains text for each speech.
        text_dir (str): Path to directory where text files should be saved. Default to same
            directory as audio files.
    """

    df["lines"] = df["anftext"].apply(lambda x: sent_tokenize(x))
    df["filename_anforande_text"] = df["filename_anforande_audio"].apply(
        lambda x: Path(x).parent / f"{Path(x).stem}.txt"
    )

    for _, row in df.iterrows():
        with open(os.path.join(text_dir, row["filename_anforande_text"]), "w") as f:
            for line in row["lines"]:
                f.write(line + "\n")

    return df
