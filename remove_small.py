import os
import re

# Folder containing the files
folder_path = "testsets_small"  # Change this to your actual folder path

# Regex pattern to match filenames like anything_X_Y_Z.npy
pattern = re.compile(r"^(.+?)_(\d+)_(\d+)_(\d+)\.npy$")

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        prefix, X, Y, Z = match.groups()  # Extract components
        Y = int(Y)  # Convert Y to integer

        if Y > 10:
            # Build full file paths for both the data and label files
            file_path = os.path.join(folder_path, filename)
            label_file_path = file_path.replace(
                ".npy", "_labels.npy")  # Get corresponding label file

            # Remove both the data file and its label file
            for f in [file_path, label_file_path]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted: {f}")
