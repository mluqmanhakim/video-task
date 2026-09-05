from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np
import pandas as pd


current_dir = Path.cwd()
config_path = current_dir.parents[0] / "config" / "task1.json"
model_dir = current_dir.parents[0] / "model"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Minimum number of frames a person must remain almost stationary to be considered stopped.
STOP_FRAMES = config["stop_frames"]
# Maximum movement in pixels between frames to consider a person stationary.
STOP_DISTANCE = config["stop_distance"]
# Number of frames to remember for each person
HISTORY_LENGTH = config["history_length"]
STOP_ZONE = np.array(config["stop_zone"], dtype=np.int32)
ENTRANCE_A = config["entrance_a"]
ENTRANCE_B = config["entrance_b"]


video_path = current_dir.parents[0] / "input" / config["input_video_filename"]
output_dir = current_dir.parents[0] / "output"
output_path = output_dir / config["output_video_filename"]
output_csv_path = output_dir / config["output_csv_filename"]


print("Writing output into csv ...")
result_data = {
    "Total Interested": 1,
    "Interested Entered": 2,
    "Interested Passed By": 3
}
df = pd.DataFrame(result_data, index=[0])
df.to_csv(output_csv_path, index=False)

print(output_csv_path)

print("Done")
