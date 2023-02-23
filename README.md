# Riksdagens anföranden: audio and transcription alignment

Code, instructions and metadata for reproducing the RixVox dataset.

The entire pipeline from start to finish is untested. Start an issue if you run into any problems.

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

**Note**: The dataset took weeks to process and changes to scripts were made along the way. I haven't tried running the entire pipeline from beginnning to end. Start an issue if you run into any problems.

1. Download speech dataset from Riksdagen's öppna data. Run [`download_text_anforanden.sh`](https://github.com/kb-labb/riksdagen_anforanden/blob/main/download_text_anforanden.sh) 
  
    ```bash
    bash download_text_anforanden.sh
    ```

2. Preprocess the downloaded files to extract relevant metadata fields ([`preprocess_speeches_metadata.py`](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/preprocess_speeches_metadata.py)).

    ```bash
    python scripts/preprocess_speeches_metadata.py
    ```

3. Using the metadata from the above preprocessing step, we query a different API endpoint (https://data.riksdagen.se/api/mhs-vodapi?) to download metadata about media files associated with the speeches. We also grab the text of the speeches from this endpoint and clean it up ([download_audio_metadata.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/download_audio_metadata.py))

    ```bash
    python scripts/download_audio_metadata.py
    ```

4. Use the download links from previous step to download the audio files associated with each debate ([download_audio.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/download_audio.py))

    ```bash
    python scripts/download_audio.py
    ```

5. Convert from mp3 to wav ([mp3_to_wav.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/mp3_to_wav.py))

    ```bash
    python scripts/convert_mp3_to_wav.py
    ```

6. Run automated transcription of the debate file, do fuzzy string matching between automated transcripts and official transcripts, and run diarization (recommend to chunk your data in batches before running this). [speech_finder.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/speech_finder.py)

7. Run [diarization_text_matcher.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/diarization_text_matcher.py) to assign speeches/speakers to the diarization results, using fuzzy string matched timestamps as guide.

    ```bash
    python scripts/diarization_text_matcher.py
    ```

8. Apply some heuristic filters to filter out low confidence diarization results and strange speeches. [heuristic_filter.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/heuristic_filter.py)

    ```bash
    python scripts/heuristic_filter.py
    ```

9. Split the audio files by individual speeches. We use the available metadata, which indicates the start and end time of speeches, to split the audio file by individual speeches ([split_audio_by_speeches.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/split_audio_by_speeches.py)).
        
        ```bash
        python scripts/split_audio_by_speeches.py
        ```

10. Run Wav2Vec2 transcription on again, but this time on the speech level instead of on the debate level. [wav2vec2_transcribe.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/wav2vec2_transcribe.py).

    ```bash
    python scripts/wav2vec2_transcribe.py
    ```

11. Perform another round of fuzzy string matching ([text_matcher.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/text_matcher.py)), this time on the speech level. We compare automated transcripts of the speech segmentations with the official transcripts. These are later used for further quality filtering, as it helps us determine the overlap between the speech segments and official transcripts.

    ```bash
    python scripts/text_matcher.py
    ```
12. If you want to recreate RixVox, perform further quality filtering by running [rixvox_filter.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/rixvox_filter.py).
    
        ```bash
        python scripts/rixvox_filter.py
        ```

13. Split the text of official transcripts in to sentences and output to newline separated text files ([split_text_by_speeches.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/split_text_by_speeches.py).

    ```bash
    python scripts/split_text_by_speeches.py
    ```

14. Use `aeneas` library to force align transcripts and audio ([force_align_audio.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/force_align_audio.py)).

    ```bash
    python scripts/force_align_audio.py
    ```

15. We now create audio snippets shorter than 30s for the final RixVox dataset. Combine/append sentences up to 30 seconds in length. We split those combined segments out of the speech audio files. This output becomes the final RixVox dataset. ([rixvox_splits.py](https://github.com/kb-labb/riksdagen_anforanden/blob/main/scripts/rixvox_splits.py))

    ```bash
    python scripts/rixvox_splits.py
    ```
