# Riksdagens anföranden: audio and transcription alignment

## Environment

Install [aeneas dependencies](https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md) (Linux):

```bash
wget https://raw.githubusercontent.com/readbeyond/aeneas/master/install_dependencies.sh
bash install_dependencies.sh
```

Install conda environment named `audio` using [environment.yml](https://github.com/kb-labb/riksdagen_anforanden/blob/main/environment.yml) file:

```bash
conda env create -f environment.yml
```

## Scripts

Download speech dataset from Riksdagen's öppna data. Run [`download_text_anforanden.sh`](https://github.com/kb-labb/riksdagen_anforanden/blob/main/download_text_anforanden.sh)

```bash
bash download_text_anforanden.sh
```

Preprocess the downloaded files to extract relevant metadata fields ([`preprocess_speeches_metadata.py`](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/preprocess_speeches_metadata.py)).

```bash
python scripts/preprocess_speeches_metadata.py
```

Using the metadata from the above preprocessing step, we query a different API endpoint (https://data.riksdagen.se/api/mhs-vodapi?) to download metadata about media files associated with the speeches. We also grab the text of the speeches from this endpoint and clean it up ([download_audio_metadata.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/download_audio_metadata.py))

```bash
python scripts/download_audio_metatdata.py
```

Use the download links from previous step to download the actual audio files associated with each debate ([download_audio.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/download_audio.py))

```bash
python scripts/download_audio.py
```

The audio files we've downloaded cover entire debates, sometimes spanning over multiple motioner, betänkanden, interpellationer. We use the available metadata, which indicates the start time and the duration of speeches, to split the audio file by individual speeches ([split_audio_by_speeches.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/split_audio_by_speeches.py))

```bash
python scripts/split_audio_by_speeches.py
```

We repeat the process for the speech text transcripts. A sentence tokenized text file is created for each individual speech, with newline separated sentences (one sentence per line). If the split audio file is found under `data/audio/GS01TU11/2442210030027487721_aud_0_166.mp3`, then the text file will be created as `data/audio/GS01TU11/2442210030027487721_aud_0_166.txt`. See [split_text_by_speeches.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/split_text_by_speeches.py).

```bash
python scripts/split_text_by_speeches.py
```

Use `aeneas` library to force align transcripts and audio ([force_align_audio.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/force_align_audio.py)). 

```bash
python scripts/force_align_audio.py
```