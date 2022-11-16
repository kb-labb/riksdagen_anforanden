import librosa
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline


class AnforandeDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        len(self.df)

    def __getitem__(self, idx):
        audio_filepath = "data/audio/" + df["filename_anforande_audio"].iloc[idx]
        print(audio_filepath)
        speech_array, sampling_rate = torchaudio.load(audio_filepath)
        speech_array = torch.mean(speech_array, dim=0)

        # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline.__call__.inputs
        # dict form input
        inputs = {"sampling_rate": sampling_rate, "raw": speech_array.numpy()}
        return inputs


class DiarizationDataset(Dataset):
    def __init__(self, metadata_dict):
        self.filepaths = metadata_dict["filename_anforande_audio"]
        self.dokid = metadata_dict["dokid"]
        self.anforande_nummer = metadata_dict["anforande_nummer"]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio_filepath = "data/audio/" + self.filepaths[idx]
        # speech_array, sample_rate = librosa.load(audio_filepath, sr=16000)
        speech_array, sample_rate = torchaudio.load(audio_filepath)
        speech_array = torch.mean(speech_array, dim=0)

        # pyannote.audio pipeline expects a dict with keys "waveform" and "sampling_rate"
        inputs = {
            "sample_rate": sample_rate,
            "waveform": speech_array.to("cuda"),
            "dokid": self.dokid[idx],
            "anforande_nummer": self.anforande_nummer[idx],
            "filename_anforande_audio": self.filepaths[idx],
        }
        return inputs


if __name__ == "__main__":
    pipe = pipeline(model="KBLab/wav2vec2-large-voxrex-swedish", device=0)
    df = pd.read_parquet("data/df_audio_metadata.parquet")
    df = df.reset_index(drop=True)
    anforande_dataset = AnforandeDataset(df)
    batch_size = 2

    texts = []
    for out in tqdm(
        pipe(anforande_dataset, batch_size=batch_size, chunk_length_s=30, stride_length_s=5),
        total=len(df) // batch_size,
    ):
        texts.append(out)
