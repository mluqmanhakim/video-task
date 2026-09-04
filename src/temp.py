from pathlib import Path

current_dir = Path.cwd()
data_dir = current_dir.parents[0] / "data"
video_path = data_dir / "enter.mp4"


output_dir = data_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)



print(output_dir)