import cv2
import os


def detect_edges_in_folder(input_folder, output_folder,
                           threshold1=100, threshold2=200):
    """
    Apply Canny edge detection to all images in a folder

    Parameters:
    - input_folder: Path to folder containing images
    - output_folder: Path to save edge-detected images
    - threshold1: First threshold for hysteresis procedure
    - threshold2: Second threshold for hysteresis procedure
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for image_file in image_files:
        # Read image
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Could not read image: {image_file}")
            continue

        # Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(img, threshold1, threshold2)
        # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        # edges = sobelx  # - sobely

        # Save the result
        output_path = os.path.join(output_folder, f"edges_{image_file}")
        cv2.imwrite(output_path, edges)

        print(f"Processed: {image_file}")


# Example usage
input_folder = "./CroppedYalePNG/"
output_folder = "./output/"
detect_edges_in_folder(input_folder, output_folder)
