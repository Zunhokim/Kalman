import os

# Parameters
folder_path = '/Users/junho_kim/Downloads/Prediction_path_image_2/tenAlpha'  # Folder containing the files
prefix_to_remove = '10_blending_'  # Prefix to remove from file names
new_prefix = 'ten_shapes_'  # New prefix to replace the old one

# List all files in the folder
files = os.listdir(folder_path)

# Process each file
for file_name in files:
    # Check if the file name starts with the specified prefix
    if file_name.startswith(prefix_to_remove):
        # Replace the prefix
        new_name = new_prefix + file_name[len(prefix_to_remove):]
        # Generate full file paths
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: "{file_name}" to "{new_name}"')

print("All matching files have been renamed.")
