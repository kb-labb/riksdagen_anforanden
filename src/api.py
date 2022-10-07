import time
import requests
from json import loads


def get_audio_metadata(rel_dok_id, backoff_factor=0.2):
    """
    Download metadata for anföranden that have related media
    files at riksdagens öppna data. Those anföranden that have
    a non-empty rel_dok_id tend to be the ones that have associated
    media files.

    Args:
        rel_dok_id (str): rel_dok_id for the session from files
        in https://data.riksdagen.se/data/anforanden/.
        backoff_factor (int): Slow down request frequency if riksdagen's
        API rejects requests.

    Returns:
        dict: Nested metadata fields with transcribed texts, media file
        URLs and more.
    """
    base_url = "https://data.riksdagen.se/api/mhs-vodapi?"

    for i in range(3):
        backoff_time = backoff_factor * (2**i)
        anforande_metadata = requests.get(f"{base_url}{rel_dok_id}")

        if anforande_metadata.status_code == 200:
            anforande_metadata = loads(anforande_metadata.text)
            return anforande_metadata

        else:
            print(f"rel_dok_id {rel_dok_id} failed with code {anforande_metadata.status_code}")

    time.sleep(backoff_time)


def get_audio_file(audiofileurl, backoff_factor=0.2):
    """
    Download mp3 files from riksdagens öppna data.
    Endpoint https://data.riksdagen.se/api/mhs-vodapi?

    Args:
        audiofileurl (str): Download URL for the mp3 file.
        E.g: https://mhdownload.riksdagen.se/VOD1/PAL169/2442205160012270021_aud.mp3
        backoff_factor (int): Slow down request frequency if riksdagen's
        API rejects requests.

    Returns
    """

    for i in range(3):
        backoff_time = backoff_factor * (2**i)
        anforanden_media = requests.get(f"{audiofileurl}")
