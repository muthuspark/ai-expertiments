import os

import cv2

from utilities.toolz import get_file_name


def extract_images_from_video(video_path):
    filename = get_file_name(video_path)
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video!")
        exit(1)

    # Define a frame counter
    frame_count = 0
    frames_input_folder = os.path.join("input", filename)
    os.makedirs(frames_input_folder, exist_ok=True)  # Create directory if it doesn't exist

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("No more frames to capture!")
            break

        # Save the frame as a JPG image with a filename based on frame count
        filepath = os.path.join(frames_input_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(filepath, frame, params=[int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Increase frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_count} frames and saved them to {frames_input_folder}")

    return frames_input_folder
