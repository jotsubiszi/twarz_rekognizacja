import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar
import tkinter.filedialog as fd


def sliding_probability_match(main_photo, template, stride=1):
    """
    Compare template across main photo using sliding window
    where template's gray values represent probabilities (0-255 as 0%-100%)

    Args:
        main_photo: Edge-detected image (larger than template)
        template: Blurred edge template (smaller than main photo)
        stride: Step size for sliding window (pixels)

    Returns:
        best_match_val: Highest probability score found
        best_match_loc: (x,y) position of best match
        probability_map: Heatmap of all probabilities
    """
    # Convert to float and normalize
    main_photo = main_photo.astype(np.float32) / 255.0
    template = template.astype(np.float32) / 255.0

    # Get dimensions
    h, w = main_photo.shape
    t_h, t_w = template.shape

    # Initialize probability map
    prob_map = np.zeros((h - t_h + 1, w - t_w + 1))

    # Slide template across main image
    best_match_val = -1
    best_match_loc = (0, 0)

    for y in tqdm(range(0, h - t_h + 1, stride)):
        for x in range(0, w - t_w + 1, stride):
            # Extract current window
            window = main_photo[y:y+t_h, x:x+t_w]

            # Calculate probability score
            prob_score = np.sum(window * template)  # / np.sum(window)

            # Store in probability map
            prob_map[y, x] = prob_score

            # Track best match
            if prob_score > best_match_val:
                best_match_val = prob_score
                best_match_loc = (x, y)

    return best_match_val, best_match_loc, prob_map


def prepare_image_for_face_recognition(img, template_width=100, template_height=100, max_angle_deviation=20):
    """
    Prepares an image for face recognition with improved ellipse filtering and scaling.

    Args:
        img: Input BGR image
        template_width: Width of the reference template (default 100)
        template_height: Height of the reference template (default 100)
        max_angle_deviation: Maximum allowed deviation from vertical orientation in degrees (default 20)

    Returns:
        mask: Black image with white ellipses where valid face regions were detected
        scales: List of scaling factors for each valid ellipse
        valid_ellipses: List of valid ellipses (center, axes, angle)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define skin color ranges in HSV
    lower_skin1 = np.array([0, 30, 30], dtype=np.uint8)
    upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 30, 30], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get skin colors
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of skin regions
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create black mask of same size as original image
    mask = np.zeros_like(skin_mask)
    scales = []
    valid_ellipses = []

    # Calculate template aspect ratio
    template_aspect = template_height / template_width

    for contour in contours:
        # Skip small contours that are likely noise
        if cv2.contourArea(contour) < 1000:
            continue

        # Fit ellipse to contour
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)

            # Check orientation (should be within max_angle_deviation of vertical)
            # OpenCV's angle is 0-180 degrees, where 0 means vertical
            angle_from_vertical = min(angle, 180 - angle)
            if angle_from_vertical > max_angle_deviation:
                continue

            # Calculate ellipse aspect ratio
            ellipse_aspect = major_axis / minor_axis

            # Determine which dimension to use for scaling
            if ellipse_aspect > template_aspect:
                # Ellipse is more oblong than template - scale to match width
                scale = minor_axis / template_width
            else:
                # Ellipse is less oblong than template - scale to match height
                scale = major_axis / template_height

            # Draw ellipse on mask
            cv2.ellipse(mask, ellipse, 1, -1)

            # Store results
            scales.append(scale)
            valid_ellipses.append(ellipse)

    return mask, scales, valid_ellipses
# Example usage:
# img = cv2.imread('person.jpg')
# mask, scales = prepare_image_for_face_recognition(img)
# cv2.imshow('Face Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img_path = fd.askopenfilename()
if img_path == ():
    print("Error: invalid file")
    exit(1)
# Load images
main_photo = cv2.imread(img_path)
template = cv2.imread(
    './templates/cropped/cropped_average_blurred_image.png', cv2.IMREAD_GRAYSCALE)
# Run matching
template_h, template_w = template.shape[:2]
mask, scales, _ = prepare_image_for_face_recognition(
    main_photo, template_w, template_h)
main_photo = cv2.cvtColor(main_photo, cv2.COLOR_BGR2GRAY)
main_photo = np.multiply(cv2.Canny(main_photo, 128, 255), mask)
best_score, best_loc, heatmap = sliding_probability_match(main_photo, template)
# OR for faster results:
# best_score, best_loc, heatmap = optimized_sliding_match(main_photo, template)

print(f"Best match probability: {best_score:.2%} at position {best_loc}")

# Visualize
heatmap_vis = cv2.normalize(heatmap, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
# heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

# Draw rectangle around best match
x, y = best_loc
t_h, t_w = template.shape

plt.subplot(121)
plt.imshow(mask)

plt.subplot(122)
cv2.circle(main_photo, (x + 84, y + 96), 150, (255, 255, 255), 2)
plt.imshow(main_photo)
plt.show()
