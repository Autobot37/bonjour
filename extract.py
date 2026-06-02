import os
import shutil
import time

dataset_dir = r"C:\Users\bagri\Downloads\e88186124ec611f1\dataset"
archive_dir = os.path.join(dataset_dir, "archive")

# Ensure archive directory exists
os.makedirs(archive_dir, exist_ok=True)

# List all files in dataset directory and move csv files (except train.csv and test.csv)
for file in os.listdir(dataset_dir):
    if file.endswith(".csv") and file not in ["train.csv", "test.csv", "real_test.csv"]:
        src_path = os.path.join(dataset_dir, file)
        dst_path = os.path.join(archive_dir, file)
        shutil.move(src_path, dst_path)
        print(f"Moved: {file}")

# Move submission.csv from archive to dataset if modified within 10 minutes
submission_archive = os.path.join(archive_dir, "submission.csv")
if os.path.exists(submission_archive):
    mtime = os.path.getmtime(submission_archive)
    if time.time() - mtime <= 600:  # 10 minutes = 600 seconds
        shutil.move(submission_archive, os.path.join(dataset_dir, "submission.csv"))
        print("Moved submission.csv from archive to dataset/ (modified within 10 minutes)")

