import torch
import torchaudio
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from torch.utils.data import Dataset


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
