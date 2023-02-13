import os
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pydub import AudioSegment
from nltk import sent_tokenize


def split_audio_by_speech(df, audio_dir="data/audio", dest_dir="data/audio2", file_exists_check=True):
    """
    Split audio file by anförande (speech) and save to disk in folder for specific dokid.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            df["filename"] looks like "H901KrU5/2442204200009516121_aud.mp3",
            i.e. {dokid}/{filename}.
        audio_dir (str): Path to directory where audio files should be saved.
        file_exists_check (bool): If True, checks whether split file already exists and
            skips it. When False, reprocesses all files.
    """

    if pd.isnull(df["start_adjusted"].iloc[0]):
        return None

    filename_dokid = df["filename"].iloc[0]
    segments = df[["start_adjusted", "end_adjusted"]].to_dict(orient="records")
    sound = AudioSegment.from_mp3(os.path.join(audio_dir, filename_dokid))
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)

    os.makedirs(os.path.join(dest_dir, Path(filename_dokid).parent), exist_ok=True)

    for segment in segments:
        if pd.isnull(segment["start_adjusted"]) or pd.isnull(segment["end_adjusted"]):
            continue

        start = float(segment["start_adjusted"]) * 1000  # ms
        end = float(segment["end_adjusted"]) * 1000
        split = sound[start:end]

        filename = Path(filename_dokid).parent / Path(filename_dokid).stem  # Filename without extension.

        filename_speech = Path(f"{filename}_{segment['start_adjusted']}_{segment['end_adjusted']}.wav")

        if file_exists_check:
            if os.path.exists(os.path.join(audio_dir, filename_speech)) or os.path.exists(
                os.path.join(dest_dir, filename_speech)
            ):
                print(f"File {filename_speech} already exists.")
                continue

        split.export(os.path.join(dest_dir, filename_speech), format="wav")

    print(f"{filename_speech.parent} complete", end="\r", flush=True)
    return None


def transcribe(
    df, pipe, folder="data/audio", chunk_length_s=50, stride_length_s=7, return_timestamps="word", full_debate=False
):
    """
    Transcribe audio files in a dataframe using the wav2vec2 model.

    Args:
        df (pd.DataFrame): Dataframe with either columns "filename" or "filename_anforande_audio"
            depending on if the full debate or only the speech is transcribed.
        pipe (transformers.pipeline): Wav2vec2 pipeline.
        folder (str): Path to folder with audio files.
        chunk_length_s (int): W2V chunks long files. Length of each chunk in seconds.
        stride_length_s (int): How many second should the next chunk overlap with the previous
            chunk, to provide context.
        return_timestamps (str): Whether to return timestamps for each word.
        full_debate (bool): Whether to transcribe the full debate or only the speech.
            Needs column "filename" for full debate and "filename_anforande_audio" for speech.

    Returns:
        pd.DataFrame: Dataframe with the transcribed text and metadata.
    """

    texts = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_filepath = os.path.join(folder, row.filename_anforande_audio)

            text = pipe(
                audio_filepath,
                chunk_length_s=chunk_length_s,
                stride_length_s=stride_length_s,
                return_timestamps=return_timestamps,
            )
            text["dokid"] = row.dokid
            text["anforande_nummer"] = row.anforande_nummer
            text["start"] = row.start
            text["duration"] = row.duration

            if full_debate:
                texts_dict["filename"] = row.filename
            else:
                texts_dict["filename_anforande_audio"] = row.filename_anforande_audio
            texts.append(text)
        except Exception as e:
            print(e)

            texts_dict = {
                "text": None,
                "dokid": row.dokid,
                "anforande_nummer": row.anforande_nummer,
                "start": row.start,
                "duration": row.duration,
            }

            if full_debate:
                texts_dict["filename"] = row.filename
            else:
                texts_dict["filename_anforande_audio"] = row.filename_anforande_audio

            texts.append(texts_dict)

    df_inference = pd.DataFrame(texts)
    return df_inference


def diarize(pipe, diarization_dataset, batch_size=1, num_workers=24, prefetch_factor=4):
    """
    Diarize audio files and return a DataFrame with speaker segments.

    Parameters:
        pipe (torch.nn.Module): Diarization model.
        diarization_dataset (torch.utils.data.Dataset): Dataset with audio files to diarize.
            Should return a dictionary with keys "waveform", "sample_rate", "filename", "dokid",
            and "anforande_nummer".
        batch_size (int): Batch size for diarization. Keep at 1.
        num_workers (int | None): Number of workers for diarization.
        prefetch_factor (int): Prefetch factor for diarization. How many audio files to load
            in advance.
    """

    diarization_loader = DataLoader(
        diarization_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    speakers = []
    try:
        for index, batch in tqdm(enumerate(diarization_loader), total=len(diarization_loader)):
            for i in range(0, len(batch["waveform"])):
                try:
                    diarization = pipe(
                        {
                            "sample_rate": int(batch["sample_rate"][i]),
                            "waveform": batch["waveform"][i].to("cuda").unsqueeze(0),
                        }
                    )

                    df_speaker = pd.DataFrame(
                        [
                            {"start": segment.start, "end": segment.end, "label": label}
                            for segment, _, label in diarization.itertracks(yield_label=True)
                        ]
                    )
                    df_speaker["dokid"] = batch["dokid"][i]
                    df_speaker["anforande_nummer"] = int(batch["anforande_nummer"][i])

                    if diarization_dataset.full_debate:
                        # Save filename of full debate audio file.
                        df_speaker["filename"] = batch["filename"][i]
                    else:
                        # Save filename of speech audio file.
                        df_speaker["filename_anforande_audio"] = batch["filename"][i]
                    speakers.append(df_speaker)
                except Exception as e:
                    print(e)
                    print(batch["filename"][i])
                    df_speaker = pd.DataFrame(
                        [
                            {
                                "start": None,
                                "end": None,
                                "label": batch["filename"][i],
                                "dokid": batch["dokid"][i],
                                "anforande_nummer": int(batch["anforande_nummer"][i]),
                            }
                        ]
                    )

                    if diarization_dataset.full_debate:
                        df_speaker["filename"] = batch["filename"][i]
                    else:
                        df_speaker["filename_anforande_audio"] = batch["filename"][i]

                    speakers.append(df_speaker)
    except Exception as e:
        # In case of error it should still go ahead and return the results that
        # were successfully processed.
        print(e)

    df_speakers = pd.concat(speakers).reset_index(drop=True)
    return df_speakers


def convert_mp3_to_wav(filename, audio_dir="data/audio", dest_dir="data/audio", sample_rate=16000, channels=1):
    """
    Convert mp3 files to wav files.

    Parameters:
        filename (str): Relative filepath+filename of mp3 file inside the audio_dir.
            For example "H901KrU5/2442204200009516121_aud.mp3".
        audio_dir (str): Path to directory where audio files were saved.
        dest_dir (str): Path to directory where wav files should be saved.
        sample_rate (int): Sample rate of wav files.
        channels (int): Number of channels of wav files.
    """

    sound = AudioSegment.from_mp3(os.path.join(audio_dir, filename))
    sound = sound.set_frame_rate(sample_rate)
    sound = sound.set_channels(channels)
    # Create directory if it doesn't exist.
    os.makedirs(os.path.join(dest_dir, Path(filename).parent), exist_ok=True)
    sound.export(os.path.join(dest_dir, filename.replace(".mp3", ".wav")), format="wav")
    print(f"Converted {filename} to wav", end="\r", flush=True)


def get_corrupt_audio_files(df, audio_dir="data/audio", return_subset=True):
    """
    Get list of corrupt audio files that were not able to be force aligned.
    We retry those mp3 files that have no corresponding json sync file.

    Parameters:
        df (pandas.DataFrame): DataFrame with all audio metadata, including
        filenames of audio files.
        audio_dir (str): Path to directory where audio files were saved.
        return_subset (bool): If True, returns subset of df with corrupt audio files.
            If False, returns entire df with column "corrupt" indicating whether
            audio file is corrupt or not.
    """

    def json_exists(filename):
        return os.path.exists(Path(audio_dir) / Path(filename).parent / f"{Path(filename).stem}.json")

    df["corrupt"] = df["filename_anforande_audio"].apply(lambda x: not json_exists(x))

    if return_subset:
        return df[df["corrupt"]].reset_index(drop=True)
    else:
        return df


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

    df["lines"] = df["anftext"].apply(lambda x: sent_tokenize(x) if x is not None else [None])
    df["filename_anforande_text"] = df["filename_anforande_audio"].apply(
        lambda x: Path(x).parent / f"{Path(x).stem}.txt"
    )

    for _, row in df.iterrows():

        if row["lines"][0] is None:
            continue

        with open(os.path.join(text_dir, row["filename_anforande_text"]), "w") as f:
            for line in row["lines"]:
                f.write(line + "\n")

    return df


def split_audio_diarization_sweeper(
    df,
    audio_dir="data/audio",
    dest_dir="temp",
    segment="start",
    sweep_window=40,
    sweep_step=5,
):
    """
    Split the audio file around the predicted start_text_time or the end_text_time.
    Performing several splits sweeping with a 40 second window around the predicted time.

    Parameters:
        df (pandas.DataFrame): Subset of DataFrame with audio metadata for specific dokid.
            Should have columns "start_text_time" and "end_text_time".
            df["filename"] looks like "H901KrU5/2442204200009516121_aud.mp3",
            i.e. {dokid}/{filename}.
        audio_dir (str): Path to directory where full debate audio files are located.
        dest_dir (str): Path to directory where split audio files should be saved.
        segment (str): "start" or "end". Indicates whether to split around the start_text_time
            or the end_text_time.
        sweep_window (int): Number of seconds to sweep around the predicted time.
        sweep_step (int): Number of seconds to step forward/backward in the sweep.
    """

    if pd.isnull(df["start_text_time"].iloc[0]):
        return None

    filename_dokid = df["filename"].iloc[0]
    sound = AudioSegment.from_wav(os.path.join(audio_dir, filename_dokid))

    os.makedirs(os.path.join(dest_dir, Path(filename_dokid).parent), exist_ok=True)

    sweep_end = sweep_window if segment == "start" else -1
    sweep_files = []

    for _, row in df.iterrows():
        if segment == "start":
            segment_time = row["start_text_time"]
        elif segment == "end":
            segment_time = row["end_text_time"]
        else:
            raise ValueError(f"segment must be 'start' or 'end', not {segment}")

        for i in range(sweep_window // 2, sweep_end, sweep_step if segment == "start" else -sweep_step):
            # Backwards sweep
            start = max(0, int((segment_time - i) * 1000))  # ms
            end = int((segment_time + (40 - i)) * 1000)
            split = sound[start:end]

            filename = Path(filename_dokid).parent / Path(filename_dokid).stem  # Filename without extension.

            if segment == "start":
                filename_sweep = Path(f"{filename}_start_{start/1000}_{end/1000}.wav")
            elif segment == "end":
                filename_sweep = Path(f"{filename}_end_{start/1000}_{end/1000}_end.wav")

            split.export(os.path.join(dest_dir, filename_sweep), format="wav")
            print("Saved", filename_sweep, end="\r", flush=True)

            metadata = {
                "filename": str(filename_sweep),
                "start_sweep": start / 1000,
                "end_sweep": end / 1000,
                "start_text_time": row["start_text_time"],
                "end_text_time": row["end_text_time"],
                "dokid": row["dokid"],
                "anforande_nummer": row["anforande_nummer"],
            }
            sweep_files.append(metadata)

    print(f"{filename_sweep.parent} complete")
    return sweep_files
