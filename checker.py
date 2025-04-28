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
            prob_score = np.sum(window * template) / np.sum(window)

            # Store in probability map
            prob_map[y, x] = prob_score

            # Track best match
            if prob_score > best_match_val:
                best_match_val = prob_score
                best_match_loc = (x, y)

    return best_match_val, best_match_loc, prob_map


img_path = fd.askopenfilename()
if img_path == ():
    print("Error: invalid file")
    exit(1)
# Load images
main_photo = cv2.imread(
    img_path, cv2.IMREAD_GRAYSCALE)
main_photo = cv2.Canny(main_photo, 100, 200)
cv2.namedWindow("photo")
cv2.imshow("photo", main_photo)
cv2.destroyAllWindows()
template = cv2.imread(
    './templates/cropped/cropped_median_blurred_image.png', cv2.IMREAD_GRAYSCALE)

# Run matching
best_score, best_loc, heatmap = sliding_probability_match(main_photo, template)
# OR for faster results:
# best_score, best_loc, heatmap = optimized_sliding_match(main_photo, template)

print(f"Best match probability: {best_score:.2%} at position {best_loc}")

# Visualize
heatmap_vis = cv2.normalize(heatmap, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

# Draw rectangle around best match
x, y = best_loc
t_h, t_w = template.shape
# cv2.rectangle(heatmap_vis, (x, y), (x+t_w, y+t_h), (0, 255, 0), 2)

plt.subplot(121)
plt.imshow(heatmap_vis)

plt.subplot(122)
plt.imshow(main_photo)
plt.show()
