#!/usr/bin/env python3
import os
import sys
import zipfile
import kagglehub

DATASET_ID = "karkavelrajaj/amazon-sales-dataset"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'amazon_sales')


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
        destination = os.path.join(OUTPUT_DIR, os.path.basename(archive_path))
        os.replace(archive_path, destination)
        print(f"Moved file to: {destination}")

    print("Amazon Product Sales dataset is ready for use!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}", file=sys.stderr)
        sys.exit(1)
