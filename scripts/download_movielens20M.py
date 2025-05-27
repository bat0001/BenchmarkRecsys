#!/usr/bin/env python3
import os
import sys
import zipfile
import kagglehub
import shutil

DATASET_ID = "grouplens/movielens-1m-dataset"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'movielens1M')


def main():
    print(f"Downloading dataset {DATASET_ID}...")
    archive_path = kagglehub.dataset_download(DATASET_ID)
    print(f"Downloaded archive: {archive_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Extracting to: {OUTPUT_DIR}")

    if archive_path.lower().endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(OUTPUT_DIR)
        print("Extraction complete.")
    else:
        print(f"Copying dataset folder from {archive_path} to {OUTPUT_DIR}")
        for item in os.listdir(archive_path):
            s = os.path.join(archive_path, item)
            d = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Copy complete.")

    print("Movielens1M dataset is ready for use!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}", file=sys.stderr)
        sys.exit(1)
