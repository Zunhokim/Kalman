import cv2
import glob
import numpy as np

# Parameters
image_folder = '/Users/junho_kim/Downloads/Prediction_image'  # Path to the folder containing images
video_name = '/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/kalman_filter/output_video_predict_30fps_smooth.avi'  # Output video file path
frame_rate = 30  # Frames per second

# Get list of image files
images = sorted(glob.glob(f"{image_folder}/*.png"))
if not images:
    raise ValueError("No images found in the specified folder.")

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
    for alpha in np.linspace(0, 1, 10):  # Create intermediate frames
        # Blend the images
        smoothed_frame = cv2.addWeighted(previous_frame, 1 - alpha, frame, alpha, 0)
        video.write(smoothed_frame)
    
    previous_frame = frame

# Release the video writer
video.release()
print(f"Video saved as {video_name}")
