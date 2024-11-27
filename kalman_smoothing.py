import cv2
import glob
import os
import re
import numpy as np

# Parameters
image_folder = '/Users/junho_kim/Downloads//BE/beeline_image'  # Path to the folder containing images
video_name = '/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/kalman_filter/표면인식_직선형_8_30.avi'  # Output video file path
frame_rate = 30  # Frames per second

# Function to extract numeric parts of the filename for sorting
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the first sequence of digits in the filename
    return int(match.group()) if match else float('inf')  # Return the number or infinity if no number found

# Get list of image files and sort them by extracted number
images = glob.glob(f"{image_folder}/*.png")
images = sorted(images, key=lambda x: extract_number(os.path.basename(x)))
if not images:
    raise ValueError("No images found in the specified folder.")

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Initialize Kalman filter for smoothing
kalman = cv2.KalmanFilter(6, 3)  # 6 state variables, 3 measurements (RGB channels)
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0]], np.float32)
kalman.transitionMatrix = np.eye(6, dtype=np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03

# Process each image
previous_frame = frame
for i, image in enumerate(images):
    frame = cv2.imread(image)

    # Apply smoothing between previous and current frame
    for alpha in np.linspace(0, 1, 8):  # Create intermediate frames
        # Blend the images
        smoothed_frame = cv2.addWeighted(previous_frame, 1 - alpha, frame, alpha, 0)
        video.write(smoothed_frame)
    
    previous_frame = frame

# Release the video writer
video.release()
print(f"Video saved as {video_name}")
