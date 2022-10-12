from aeneas.executetask import ExecuteTask
from aeneas.task import Task

config_string = u"task_language=swe|is_text_type=plain|os_task_file_format=json"
task = Task(config_string=config_string)

task.__dict__
task.audio_file_path_absolute = "test.mp3"
task.text_file_path_absolute = "data/text/anf1.txt"
task.sync_map_file_path_absolute = "map3.json"

ExecuteTask(task).execute()
task.output_sync_map_file()
