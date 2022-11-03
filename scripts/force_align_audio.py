import os
import multiprocessing as mp
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.audiofile import AudioFileUnsupportedFormatError
from tqdm import tqdm


def force_align_audio_transcript(files):
    """
    Force-align audio and transcript using Aeneas.

    Parameters:
        files (list): List of tuples/lists with audio and transcript filenames.
            [[audiofile1, transcriptfile1], [audiofile2, transcriptfile2], ...].
            Can take a single pair of filenames as well.
    """
    config_string = "task_language=swe|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)

    audiofile = files[0]
    transcriptfile = files[1]

    folder = os.path.dirname(audiofile)
    filename = audiofile.rsplit("/")[-1].split(".")[0]
    json_file = os.path.abspath(os.path.join("data/audio", folder, filename + ".json"))

    if os.path.exists(json_file):
        print("File " + json_file + " already exists.", end="\r", flush=True)
        return None

    task.sync_map_file_path_absolute = json_file

    try:
        task.audio_file_path_absolute = os.path.abspath(os.path.join("data/audio", audiofile))
        task.text_file_path_absolute = os.path.abspath(os.path.join("data/audio", transcriptfile))
    except AudioFileUnsupportedFormatError as audioerror:
        print(audioerror, end="\r", flush=True)
        print("Audiofile " + audiofile + " is corrupt or not supported.", end="\r", flush=True)

    try:
        print("Aligning " + audiofile + " and " + transcriptfile, end="\r", flush=True)
        ExecuteTask(task).execute()
        task.output_sync_map_file()
    except Exception as e:
        print(e, end="\r", flush=True)
        print("Failed to align " + audiofile + " and " + transcriptfile, end="\r", flush=True)


if __name__ == "__main__":
    with open("data/speeches_files_aeneas.txt", "r") as f:
        files = f.readlines()

    files = [tuple(file.split()) for file in files]
    pool = mp.Pool(26)
    pool.map(force_align_audio_transcript, tqdm(files, total=len(files)), chunksize=4)
    pool.close()
