import os
import requests
import whisper
import pandas as pd
from stable_whisper import modify_model
from json import loads
from pprint import pprint
from tqdm import tqdm
from pydub import AudioSegment

base_url = "https://data.riksdagen.se/api/mhs-vodapi?"
anforande_id = "H510421"

anforanden_media = requests.get(f"{base_url}{anforande_id}")

audio_json = loads(anforanden_media.text)


df = pd.DataFrame(audio_json["videodata"])
df = df.explode("speakers").reset_index(drop=True)
df_files = pd.json_normalize(df["streams"], ["files"])
df_speakers = pd.json_normalize(df["speakers"])
df = df.drop(columns=["streams", "speakers"])
df = pd.concat([df, df_files], axis=1)
df = pd.concat([df, df_speakers], axis=1)

df = df[
    [
        "dokid",
        "party",
        "start",
        "duration",
        "debateseconds",
        "title",
        "text",
        "debatename",
        "debatedate",
        "url",
        "debateurl",
        "id",
        "subid",
        "audiofileurl",
        "downloadfileurl",
        "debatetype",
        "number",
        "anftext",
    ]
]

df.columns


url = df["audiofileurl"][0]
response = requests.get(url)
file_path = os.path.join("data", "data_audio", url.rsplit("/")[-1])

with open(file_path, "wb") as f:
    f.write(response.content)


model = whisper.load_model("large", "cuda")
result = model.transcribe(file_path)
sound = AudioSegment.from_mp3("data_audio/2442205160012270021_aud.mp3")

split = sound[result["segments"][2]["start"] * 1000 : result["segments"][2]["end"] * 1000]
split.export("test.mp3", format="mp3")


result["segments"][3]["text"]
result["segments"][3]["word_timestamps"]

from fuzzysearch import find_near_matches

large_string = df["anftext"][8]
query_string = result["segments"][223]["text"]
find_near_matches(query_string, large_string, max_l_dist=18)
query_string

result["segments"][168:172]
