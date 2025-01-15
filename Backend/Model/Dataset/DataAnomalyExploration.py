import shutil
import os

# Path to the text file containing file names
file_list_path = "C:\\Users\\Mehdi\\Desktop\\TrafficSignsProject-main\\TrafficSignsProject-main\\file_names.txt"
# Source folder where images are located (Dataset folder)
source_folder = "C:\\Users\\Mehdi\\Desktop\\TrafficSignsProject-main\\TrafficSignsProject-main\\Dataset"
# Destination folder
destination_folder = "output"

# Read file names from the text file
with open(file_list_path, "r") as f:
    file_names = [line.strip().strip('"') for line in f]  # Remove extra quotes

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Copy files
for file_name in file_names:
    found = False  # Flag to check if the file is found
    # Check both .jpg and .png extensions
    extensions = [".jpg", ".png"]
    
    # Walk through all subdirectories of source_folder
    for root, dirs, files in os.walk(source_folder):
        for ext in extensions:
            if file_name + ext in files:  # Check if the file with the extension exists
                source_file = os.path.join(root, file_name + ext)

                shutil.copy(source_file, destination_folder)
                print(f"Copied: {file_name}{ext}")
                found = True
                break  # Stop after finding the first matching file
        if found:
            break  # Stop searching in other directories if file is found
    
    if not found:
        print(f"File not found: {file_name} (for both .jpg and .png)")
