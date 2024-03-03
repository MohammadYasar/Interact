import cv2
import os
import re  # Import the re module for alphanumeric sorting

# Directory containing the PNG files
image_folder = '/home/msy9an/TROExperiments/Skeletons'

# Output video filename
video_name = 'output_video.mp4'

# Frame rate of the output video (frames per second)
frame_rate = 30

# Function to sort the PNG files in alphanumeric order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', data)]

# Get the list of PNG files in the directory and sort them
images = [img for img in sorted(os.listdir(image_folder), key=sorted_alphanumeric) if img.endswith(".png")]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the VideoWriter and close the video file
video.release()

print(f"Video '{video_name}' created successfully.")
