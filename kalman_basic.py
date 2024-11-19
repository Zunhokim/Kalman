import cv2
import glob
import os
import re

# Parameters
image_folder = '/Users/junho_kim/Downloads/SE_Prediction_path_image'  # Path to the folder containing images
video_name = '/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/kalman_filter/output_video_predict_5.avi'  # Output video file path
frame_rate = 5  # Frames per second

# Function to extract numeric parts of the filename for sorting
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the first sequence of digits in the filename
    return int(match.group()) if match else float('inf')  # Return the number or infinity if no number found

# Get list of image files and sort them by extracted number
images = glob.glob(f"{image_folder}/*.png")
images = sorted(images, key=lambda x: extract_number(os.path.basename(x)))

# Read the first image to get the frame size
if len(images) > 0:
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
else:
    raise ValueError("No images found in the specified folder!")

# Define video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Write each image to the video
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release the video writer
video.release()
print(f"Video saved as {video_name}")
