import cv2
import glob

# Parameters
image_folder = '/Users/junho_kim/Downloads/Prediction_image_1'  # Path to the folder containing images
video_name = '/Users/junho_kim/Desktop/zunobono/2024_2학기/캡디/kalman_filter/output_video_predict.avi'  # Output video file path
frame_rate = 20  # Frames per second

# Get list of image files
images = sorted(glob.glob(f"{image_folder}/*.png"))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

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