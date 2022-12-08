import multiprocessing as mp

import librosa
import pandas as pd
import torch
import torchaudio
from pyannote.audio import Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

df = pd.read_parquet("data/df_audio_metadata.parquet").reset_index(drop=True)
# df["filename_anforande_mp3"] = df["filename_anforande_audio"].str[:-3] + "mp3"


def check_valid_audio_files(filename):
    valid_audio = []

    audio_filepath = "data/audio/" + filename

    try:
        speech_array, sampling_rate = torchaudio.load(audio_filepath)
        speech_array = torch.Tensor(speech_array)
        if speech_array.dim() >= 2:
            speech_array = torch.mean(speech_array, dim=0)

        # print(speech_array)
        # print(audio_filepath)
        if speech_array.shape[0] > 0:
            valid_audio.append(True)
        else:
            valid_audio.append(False)

    except Exception as e:
        # print(e)
        # print(audio_filepath)
        valid_audio.append(False)
    return valid_audio


with mp.Pool(26) as p:
    valid_audio = p.map(
        check_valid_audio_files,
        tqdm(df["filename_anforande_audio"].tolist(), total=len(df)),
        chunksize=4,
    )

df["valid_audio"] = valid_audio
df["valid_audio"] = df["valid_audio"].str[0]
df.to_parquet("data/df_audio_metadata.parquet")
