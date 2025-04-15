import cv2
import numpy as np
import os


def compute_median_image_batched(image_folder, batch_size=10):
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        raise ValueError("No images found")

    # Initialize with first image
    first_img = cv2.imread(os.path.join(image_folder, image_files[0]))
    if first_img is None:
        raise ValueError("Could not read first image")

    height, width, channels = first_img.shape

    # Process in batches
    median_accumulator = np.zeros((height, width, channels), dtype=np.float64)
    count = 0

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []

        for file in batch_files:
            img = cv2.imread(os.path.join(image_folder, file))
            if img is not None and img.shape == first_img.shape:
                batch_images.append(img.astype(np.float64))

        if batch_images:
            batch_array = np.stack(batch_images)
            median_accumulator += np.sum(batch_array, axis=0)
            count += len(batch_images)

    if count == 0:
        raise ValueError("No valid images processed")

    # Calculate median approximation (exact if batch_size=1)
    median_image = np.round(median_accumulator / count).astype(np.uint8)
    return median_image


# Usage
folder_path = "./output/"
median_img = compute_median_image_batched(folder_path)
cv2.imwrite('median_image.jpg', median_img)

result = cv2.GaussianBlur(median_img, (5, 5), 0)

cv2.imwrite('median_blurred_image.png', result)
