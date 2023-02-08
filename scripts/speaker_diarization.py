import gc
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from pyannote.audio import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import DiarizationDataset

df = pd.read_parquet("data/df_audio_metadata.parquet").reset_index(drop=True)
df = df[df["valid_audio"]].reset_index(drop=True)
df = df[(df["debatedate"].dt.year >= 2019) & (df["debatedate"].dt.year <= 2022)].reset_index(drop=True)


def custom_collate_fn(data):
    """
    To load batches we override the default collate_fn because the default one
    doesn't allow variable length sequences.
    In our inference step, we process one sequence at a time anyway, so it
    doesn't matter if the sequences are of different lengths."""

    waveform = [sample["waveform"] for sample in data]
    sample_rate = [torch.tensor(sample["sample_rate"]) for sample in data]
    dokid = [sample["dokid"] for sample in data]
    anforande_nummer = [sample["anforande_nummer"] for sample in data]
    filename_anforande_audio = [sample["filename_anforande_audio"] for sample in data]

    sample_rate = torch.stack(sample_rate)  # List of B 1-length vectors to single vector of dimension B
    dokid = dokid
    anforande_nummer = anforande_nummer
    filename_anforande_audio = filename_anforande_audio

    batch = {
        "sample_rate": sample_rate,
        "waveform": waveform,  # samples may have different dimensions
        "dokid": dokid,
        "anforande_nummer": anforande_nummer,
        "filename_anforande_audio": filename_anforande_audio,
    }

    return batch


df = df[["dokid", "anforande_nummer", "filename_anforande_audio"]]
diarization = DiarizationDataset(df, full_debate=True, folder="data/audio")
diarization_loader = DataLoader(
    diarization,
    batch_size=1,
    shuffle=False,
    num_workers=28,
    # collate_fn=custom_collate_fn,
    prefetch_factor=4,
)

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)

speakers = []
for index, batch in tqdm(enumerate(diarization_loader), total=len(diarization_loader)):

    audio_filepath = [os.path.join("data/audio", filename) for filename in batch["filename"]]

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
            speakers.append(df_speaker)
        except Exception as e:
            print(e)
            print(batch["filename_anforande_audio"][i])

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

            speakers.append(df_speaker)

df_speakers = pd.concat(speakers).reset_index(drop=True)
df_speakers.to_parquet("data/df_speakers_2019_2022.parquet")
