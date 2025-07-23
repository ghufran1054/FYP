# import os
# import subprocess

# # Input and output folder
# input_folder = "testing"
# output_folder = "testing_24FPS"

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Get list of all video files in the input folder
# video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# # Process each video
# for video in video_files:
#     input_path = os.path.join(input_folder, video)
#     output_path = os.path.join(output_folder, f"{video}")

#     # FFmpeg command to change FPS to 24
#     cmd = [
#         "ffmpeg", "-i", input_path,
#         "-filter:v", "fps=24",
#         "-c:a", "copy",  # Copy audio to avoid desync
#         output_path
#     ]

#     print(f"Processing: {video}")
#     subprocess.run(cmd)

# print("All videos processed!")

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress bar

# Input and output folder
input_folder = "test"
output_folder = "test_24FPS"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

def process_video(video):
    input_path = os.path.join(input_folder, video)
    output_path = os.path.join(output_folder, f"{video}")

    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Skipping (already exists): {video}")
        return

    # FFmpeg command to change FPS to 24 (unchanged)
    cmd = [
        "ffmpeg", "-i", input_path,
        "-filter:v", "fps=24",
        "-c:a", "copy",  # Copy audio to avoid desync
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL)
    print(f"Finished: {video}")

# Use ThreadPoolExecutor to process videos in parallel
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    # Wrap executor.map with tqdm for progress bar
    list(tqdm(executor.map(process_video, video_files), total=len(video_files), desc="Processing Videos"))

print("All videos processed!")