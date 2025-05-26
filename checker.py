import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter.filedialog as fd


def sliding_probability_match(main_photo, template, scales, stride=1):
    """
    Compare template across main photo using sliding window
    where template's gray values represent probabilities (0-255 as 0%-100%)

    Args:
        main_photo: Edge-detected image (larger than template)
        template: Blurred edge template (smaller than main photo)
        scales: List of scale factors to try
        stride: Step size for sliding window

    Returns:
        best_match_val: Highest probability score
        best_match_loc: (x,y) position of match in local coords
        probability_map: Full match heatmap (optional)
        best_scale: Scale factor of the best match
    """
    main_photo = main_photo.astype(np.float32) / 255.0
    template = template.astype(np.float32) / 255.0

    h, w = main_photo.shape[:2]
    t_h, t_w = template.shape[:2]

    best_match_val = -1
    best_match_loc = (0, 0)
    best_scale = 1

    for s in scales:
        resized_template = cv2.resize(template, (int(t_w * s), int(t_h * s)))
        rt_h, rt_w = resized_template.shape

        if h < rt_h or w < rt_w:
            continue  # Skip if template doesn't fit

        for y in range(0, h - rt_h + 1, stride):
            for x in range(0, w - rt_w + 1, stride):
                window = main_photo[y:y+rt_h, x:x+rt_w]
                score = np.sum(window * resized_template)

                if score > best_match_val:
                    best_match_val = score
                    best_match_loc = (x, y)
                    best_scale = s

    return best_match_val, best_match_loc, None, best_scale


def prepare_image_for_face_recognition(img, template_width=100, template_height=100, max_angle_deviation=20):
    """
    Detects possible face regions by combining HSV, YCbCr, and RGB skin color detection,
    and filters for plausible face ellipses.
    """
    # Contrast enhancement
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    img_clahe = cv2.merge((y_eq, cr, cb))
    img_bgr_eq = cv2.cvtColor(img_clahe, cv2.COLOR_YCrCb2BGR)

    hsv = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2YCrCb)
    rgb = img_bgr_eq

    # --- HSV Mask ---
    lower_hsv = np.array([0, 40, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # --- YCbCr Mask ---
    lower_ycbcr = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycbcr = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycbcr = cv2.inRange(ycbcr, lower_ycbcr, upper_ycbcr)

    # --- RGB Mask ---
    R = rgb[:, :, 2]
    G = rgb[:, :, 1]
    B = rgb[:, :, 0]

    cond_rgb = (
        (R > 60) & (G > 30) & (B > 15) &
        ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) > 10) &
        (np.abs(R - G) > 10) & (R > B)
    )
    mask_rgb = (cond_rgb.astype(np.uint8)) * 255
    
    # Convert masks to binary
    m1 = (mask_hsv > 0).astype(np.uint8)
    m2 = (mask_ycbcr > 0).astype(np.uint8)
    m3 = (mask_rgb > 0).astype(np.uint8)

    # Majority vote: keep pixel if detected in at least 2 masks
    combined = m1 + m2 + m3
    skin_mask = np.where(combined >= 2, 255, 0).astype(np.uint8)

    # --- Morphological Refinement ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Contour + Ellipse Filtering ---
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(skin_mask)
    scales = []
    valid_ellipses = []

    template_aspect = template_height / template_width

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)

            if minor_axis < 30 or major_axis < 30:
                continue
            if min(angle, 180 - angle) > max_angle_deviation:
                continue

            aspect_ratio = major_axis / minor_axis
            if aspect_ratio < 0.4 or aspect_ratio > 2.8:
                continue

            scale = minor_axis / template_width if aspect_ratio > template_aspect else major_axis / template_height

            cv2.ellipse(mask, ellipse, 255, -1)
            scales.append(scale)
            valid_ellipses.append(ellipse)

    return mask, scales, valid_ellipses
 

# Select input image
img_path = fd.askopenfilename()
if not img_path:
    print("Error: invalid file")
    exit(1)

# Load original image and grayscale template
orig_img = cv2.imread(img_path)
template = cv2.imread('./templates/cropped/cropped_average_blurred_image.png', cv2.IMREAD_GRAYSCALE)
template_h, template_w = template.shape[:2]

# Detect face candidates
mask, scales, valid_ellipses = prepare_image_for_face_recognition(orig_img, template_w, template_h, max_angle_deviation=45)

# Prepare grayscale and edges
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_img, 128, 255)
masked_edges = np.multiply(edges, mask)
masked_edges = cv2.normalize(masked_edges, None, 0, 255, cv2.NORM_MINMAX)

# Perform matching per candidate region
face_results = []
for idx, scale in enumerate(scales):
    ellipse = valid_ellipses[idx]
    (center, axes, angle) = ellipse
    cx, cy = int(center[0]), int(center[1])
    w, h = int(axes[0] * 1.5), int(axes[1] * 1.5)

    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(gray_img.shape[1], cx + w // 2), min(gray_img.shape[0], cy + h // 2)

    roi = masked_edges[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]
    roi = np.multiply(roi, roi_mask)

    best_score, best_loc, _, best_scale = sliding_probability_match(roi, template, [scale], stride=2)
    abs_x = x0 + best_loc[0]
    abs_y = y0 + best_loc[1]
    face_results.append((best_score, (abs_x, abs_y), best_scale))

# Visualize results
# Draw circles on original image
for score, loc, scale in face_results:
    x, y = loc
    radius = int(min(template_w, template_h) * scale * 0.4)
    cv2.circle(orig_img, (x + radius, y + radius), radius, (0, 255, 0), 2)

# Draw circles on edge map
for score, loc, scale in face_results:
    x, y = loc
    radius = int(min(template_w, template_h) * scale * 0.4)
    cv2.circle(masked_edges, (x + radius, y + radius), radius, 255, 1)

# Plot
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Original with Detections")
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.title("Skin Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(133)
plt.title("Edge Map with Matches")
plt.imshow(masked_edges, cmap='gray')

plt.tight_layout()
plt.show()
