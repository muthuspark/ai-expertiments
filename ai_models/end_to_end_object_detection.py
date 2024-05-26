import os.path

import cv2
import numpy as np
import requests  # For downloading remote images
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

from utilities.toolz import get_unique_color

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

label_color_map = {}


def extract_objects_from_image(_img_source, index, _output_folder, preview=False):
    # Download the image if it's a remote URL
    if _img_source.startswith("http"):
        response = requests.get(_img_source, stream=True)
        if response.status_code == 200:
            image = Image.open(response.raw)
        else:
            print(f"Error downloading image: {response.status_code}")
            exit(1)
    else:
        # Assuming local file path if not a URL
        image = Image.open(_img_source)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    label_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

        # Convert bounding box coordinates to OpenCV format
        x_min, y_min, x_max, y_max = box
        x, y = int(x_min), int(y_min)
        width, height = int(x_max - x_min), int(y_max - y_min)

        # Prepare label text
        label_text = f"{model.config.id2label[label.item()]}"  # - {round(score.item(), 3)}"

        # Draw rectangle around the detected object with the assigned color
        cv2.rectangle(label_image, (x, y), (x + width, y + height), get_unique_color(label_text), 5)

        # Calculate text placement (adjustments might be needed based on font size)
        text_width, text_height = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_offset_x = x
        text_offset_y = y - text_height - 5

        # Draw label background (optional)
        cv2.rectangle(label_image, (text_offset_x - 2, text_offset_y - 2),
                      (text_offset_x + text_width + 2, text_offset_y + text_height + 2), (255, 255, 255),
                      -1)  # Filled rectangle

        # Draw label text
        cv2.putText(label_image, label_text, (text_offset_x, text_offset_y + text_height), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)  # Black text

        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    if preview:
        # Display the labeled image
        cv2.imshow("Labeled Image", label_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    output_filename = f"labeled_image{index}.jpg"
    cv2.imwrite(os.path.join(_output_folder, output_filename), label_image)


def detect_objects_for_files_in_folder(frame_folder_name):
    output_folder = frame_folder_name.replace("input/", "output/")
    os.makedirs(output_folder, exist_ok=True)

    filenames = os.listdir(frame_folder_name)
    for i in range(0, len(filenames)):
        img_source = os.path.join(frame_folder_name, f"frame_{i}.jpg")
        extract_objects_from_image(img_source, i, output_folder)

    return output_folder
