import kagglehub
import os
import shutil
from pathlib import Path

# Specify your desired download folder
DOWNLOAD_FOLDER = "/home/rouf/data/raw/nids_dataset/H200"  # Change this to your preferred path

# Create the folder if it doesn't exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Download latest version
# Note: kagglehub downloads to its cache directory first
path = kagglehub.dataset_download("ittibydgoszcz/appraise-h2020-real-labelled-netflow-dataset")

print(f"Dataset downloaded to kagglehub cache: {path}")

# Copy files to your specified folder
destination = Path(DOWNLOAD_FOLDER)
if destination.exists():
    print(f"Destination folder already exists: {destination}")
else:
    shutil.copytree(path, destination)
    print(f"Dataset copied to: {destination}")

print(f"\nFinal location: {destination.absolute()}")