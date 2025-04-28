import cv2
import numpy as np
import os


def average_images(image_folder):
    # Get all image files from the folder
    image_files = [f for f in os.listdir(
        image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the directory")
        return None

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, channels = first_image.shape

    # Create an accumulator array with float precision
    accumulator = np.zeros((height, width, channels), np.float64)

    # Sum all images
    for image_file in image_files:
        img = cv2.imread(os.path.join(image_folder, image_file))
        if img is not None and img.shape == first_image.shape:
            accumulator += img.astype(np.float64)
        else:
            print(f"Skipping {image_file} - incompatible dimensions")

    # Calculate average
    average = accumulator / len(image_files)

    # Convert back to uint8 (0-255 range)
    average_image = np.round(average).astype(np.uint8)

    cv2.imwrite('sum_of_photos.png', accumulator)

    return average_image


# Usage
folder_path = "../photos/output/"

edge = average_images(folder_path)

cv2.imwrite('./templates/average_image.png', edge)

result = cv2.GaussianBlur(edge, (5, 5), 0)

cv2.imwrite('./templates/average_blurred_image.png', result)
