import cv2
from matplotlib import pyplot as plt
import os


def cropp_image(image, image_file):

    # Check if image loaded successfully
    if image is None:
        print("Error: Could not load image")
    else:
        # Get image dimensions (height, width)
        height, width = image.shape[:2]
        cropped_image = image
        if (height % 2) and (width % 2):
            print("Good dimensions, skipping cropping!")

        if not (height % 2):
            cropped_image = cropped_image[0:height-1, 0:width]
        if not (width % 2):
            cropped_image = cropped_image[0:height, 0:width-1]
        # Remove last column and last row
        # Syntax: image[y_start:y_end, x_start:x_end]

        # Display dimensions for verification
        print(f"Original dimensions: {width}x{height}")
        print(f"Cropped dimensions: {cropped_image.shape[1]}x{
            cropped_image.shape[0]}")

        # Save the cropped image
        output_path = os.path.join(
            './templates/cropped/', f"cropped_{image_file}")
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image: {image_file} saved successfully")


def cropp_images_in_folder():

    # Create output folder if it doesn't exist
    if not os.path.exists('./templates/cropped/'):
        os.makedirs('./templates/cropped/')

    # Get all image files
    folder = [f for f in os.listdir(
        './templates/') if f.lower().endswith(('.png'))]

    for image_file in folder:
        # Read image
        img_path = os.path.join('./templates/', image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not read image: {image_file}")
            continue

        cropp_image(img, image_file)


cropp_images_in_folder()
plt.show()
