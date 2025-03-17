import zipfile
import os

# Path to the tar file and desired extraction folder for exp1
# tar_path = os.path.expanduser("~/Downloads/imagenet-val.zip")
# extract_path = os.path.abspath("datasets/imagenet-val")

# os.makedirs(extract_path, exist_ok=True)

# # Extract all files
# with zipfile.ZipFile(tar_path, "r") as zip_ref:
#     zip_ref.extractall(extract_path)

tar_path = os.path.expanduser("~/Downloads/imagenetV2.zip")
extract_path = os.path.abspath("exp2-data/imagenetV2")

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(tar_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
