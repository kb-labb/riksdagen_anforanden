import os
import sys
from pathlib import Path

import pandas as pd
import torch
from pyannote.audio import Pipeline
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import DiarizationDataset

df = pd.read_parquet("data/df_audio_metadata.parquet").reset_index(drop=True)
df["valid_audio"] = df["valid_audio"].str[0]
df_train = df[df["valid_audio"]].reset_index(drop=True)


def custom_collate_fn(data):
    waveform = [sample["waveform"] for sample in data]
    sample_rate = [torch.tensor(sample["sample_rate"]) for sample in data]
    dokid = [sample["dokid"] for sample in data]
    anforande_nummer = [sample["anforande_nummer"] for sample in data]
    filename_anforande_audio = [sample["filename_anforande_audio"] for sample in data]

    padded_waveform = torch.nn.utils.rnn.pad_sequence(waveform, batch_first=True)
    sample_rate = torch.stack(
        sample_rate
    )  # List of B 1-length vectors to single vector of dimension B
    dokid = dokid
    anforande_nummer = anforande_nummer
    filename_anforande_audio = filename_anforande_audio

    batch = {
        "sample_rate": sample_rate,
        "waveform": padded_waveform,
        "dokid": dokid,
        "anforande_nummer": anforande_nummer,
        "filename_anforande_audio": filename_anforande_audio,
    }

    return batch


metadata_dict = df_train[["dokid", "anforande_nummer", "filename_anforande_audio"]].to_dict()
diarization = DiarizationDataset(metadata_dict)
diarization_loader = DataLoader(
    diarization,
    batch_size=1,
    shuffle=False,
    num_workers=16,
    # collate_fn=custom_collate_fn,
    prefetch_factor=2,
)

speakers = []
for index, batch in tqdm(enumerate(diarization_loader), total=len(diarization_loader)):
    try:
        audio_filepath = os.path.join("data/audio", batch["filename_anforande_audio"][0])
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)
        diarization = pipe(
            {"sample_rate": int(batch["sample_rate"][0]), "waveform": batch["waveform"]}
        )
        df_speaker = pd.DataFrame(
            [
                {"start": segment.start, "end": segment.end, "label": label}
                for segment, _, label in diarization.itertracks(yield_label=True)
            ]
        )
        df_speaker["dokid"] = batch["dokid"][0]
        df_speaker["anforande_nummer"] = int(batch["anforande_nummer"][0])
        df_speaker["filename_anforande_audio"] = batch["filename_anforande_audio"][0]
        speakers.append(df_speaker)
    except Exception as e:
        print(e)
        print(batch["filename_anforande_audio"][0])

        df_speaker = pd.DataFrame(
            [
                {
                    "start": None,
                    "end": None,
                    "label": batch["filename_anforande_audio"][0],
                    "dokid": batch["dokid"][0],
                    "anforande_nummer": batch["anforande_nummer"][0],
                    "filename_anforande_audio": batch["filename_anforande_audio"][0],
                }
            ]
        )
        speakers.append(df_speaker)

df_speakers = pd.concat(speakers).reset_index(drop=True)

df_speakers
df_speakers.to_parquet("data/df_speakers.parquet")
