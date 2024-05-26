import sys

from ai_models.end_to_end_object_detection import detect_objects_for_files_in_folder
from utilities.images_from_video import extract_images_from_video
from utilities.video_from_images import generate_video_from_frames

# should be a path passed
video_path = sys.argv[1]

# Step 1:  extract images for video
frame_folder_name = extract_images_from_video(video_path)

# Step 2: Label images
output_folder = detect_objects_for_files_in_folder(frame_folder_name)

# Step 3: Save labeled images to video
generate_video_from_frames(output_folder)
