from aeneas.exacttiming import TimeValue
from aeneas.executetask import ExecuteTask
from aeneas.language import Language
from aeneas.syncmap import SyncMapFormat
from aeneas.task import Task
from aeneas.task import TaskConfiguration
from aeneas.textfile import TextFileFormat
import aeneas.globalconstants as gc
from pydub import AudioSegment


# create Task object
config = TaskConfiguration()
config[gc.PPN_TASK_LANGUAGE] = Language.ENG
config[gc.PPN_TASK_IS_TEXT_FILE_FORMAT] = TextFileFormat.PLAIN
config[gc.PPN_TASK_OS_FILE_FORMAT] = SyncMapFormat.JSON
task = Task()
task.configuration = config
task.audio_file_path_absolute = u"/home/jonathan/voice-clips/heather-1.wav"
task.text_file_path_absolute = u"/home/jonathan/Real-Time-Voice-Cloning/demo_script.txt"

# process Task
ExecuteTask(task).execute()

full_voice_clip = AudioSegment.from_wav("/home/jonathan/voice-clips/heather-1.wav")

index = 0
for frag in task.sync_map.fragments:
    if (index != 0 and index != 6):
        print(frag.begin * 1000)
        print(frag.end * 1000)
        begin = frag.begin * 1000
        end = frag.end * 1000
        clip = full_voice_clip[float(begin): float(end)]
        clip_name = "aeneas-test-" + str(index)
        clip.export("/home/jonathan/voice-clips/aeneas_test/"+ clip_name + ".wav", format="wav")
    index = index + 1

# print(task.sync_map.fragments)

# print produced sync map
# print(task.sync_map)
