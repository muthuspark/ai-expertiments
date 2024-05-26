import os

import cv2
from PIL import Image


def generate_video_from_frames(output_folder):
    video_filename = os.path.basename(output_folder) + ".mp4"
    first_frame = os.listdir(output_folder)[0]
    image = Image.open(os.path.join(output_folder, first_frame))
    frame_width, frame_height = image.size

    # Define the video writer object with fourcc codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternatively, use 'XVID' for higher compatibility
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0,
                                   (frame_width, frame_height))  # Adjust frame rate (fps) as needed

    # Get all image filenames in sorted order (assuming consistent naming)
    image_filenames = sorted(os.listdir(output_folder), key=lambda x: int(x.split('image')[-1].split('.')[0]))

    for filename in image_filenames:
        # Construct the full image path
        image_path = os.path.join(output_folder, filename)

        # Read the image
        image = cv2.imread(image_path)

        # Resize the image if needed (ensure it matches the video frame size)
        if image.shape[1] != frame_width or image.shape[0] != frame_height:
            image = cv2.resize(image, (frame_width, frame_height))

        # Write the image to the video writer
        video_writer.write(image)

        print(f"Added image: {filename}")

    # Release the video writer
    video_writer.release()

    print(f"Created video: {video_filename}")
