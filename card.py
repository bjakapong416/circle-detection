import cv2
import numpy as np

# Dictionary of templates
mydata = {
    'redcircle': r'img/redcircle02.jpg',
    'whitecircle': r'img/whitecircle01.jpg',
    'card': r'img/sticker.jpg',
}

# Load main image
image_path = r"img\main1.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the main image.")
    exit()

# Convert to grayscale for better matching
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize main image (if necessary) to max 400px
def resize_to_max_dimension(image, max_dimension=400):
    """Resize an image to ensure its maximum dimension is less than the given size."""
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale_factor = max_dimension / max(h, w)
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        resized = cv2.resize(image, (new_width, new_height))
        return resized
    return image  # Return original if already smaller than max_dimension

# Resize main image
image = resize_to_max_dimension(image, max_dimension=400)
image_gray = resize_to_max_dimension(image_gray, max_dimension=400)

print(f"Resized Main Image: {image.shape[1]}x{image.shape[0]} pixels")

# Multi-scale template matching to detect all matches
def multi_scale_template_matching(image, template, min_scale=0.5, max_scale=2.0, scale_step=0.1, threshold=0.7):
    """Detect multiple matches of a template at multiple scales."""
    matches = []

    h_img, w_img = image.shape[:2]

    for scale in np.arange(min_scale, max_scale, scale_step):
        resized_template = cv2.resize(template, None, fx=scale, fy=scale)
        h_temp, w_temp = resized_template.shape[:2]

        if h_temp > h_img or w_temp > w_img:
            continue

        # Perform template matching
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for loc in zip(*locations[::-1]):  # Reverse to get (x, y) format
            matches.append((loc[0], loc[1], w_temp, h_temp, result[loc[1], loc[0]]))  # Add match with confidence

    return matches

# Apply Non-Maximum Suppression (NMS) to filter overlapping matches
def non_max_suppression(matches, overlap_thresh=0.3):
    """Apply Non-Maximum Suppression to filter overlapping rectangles."""
    if len(matches) == 0:
        return []

    # Extract rectangles and confidence scores
    boxes = np.array([(x, y, x + w, y + h) for (x, y, w, h, _) in matches])
    scores = np.array([conf for (_, _, _, _, conf) in matches])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[1:]]

        order = order[np.where(overlap <= overlap_thresh)[0] + 1]

    return [matches[i] for i in keep]

# Detect matches for each template
for name, path in mydata.items():
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"Error: Could not load template '{name}' from {path}")
        continue

    # Resize the template to fit within max dimensions if necessary
    template = resize_to_max_dimension(template, max_dimension=200)

    # Find all matches
    matches = multi_scale_template_matching(image_gray, template, threshold=0.7)

    # Filter matches with NMS
    filtered_matches = non_max_suppression(matches, overlap_thresh=0.3)

    if len(filtered_matches) > 0:
        print(f"Template '{name}' found {len(filtered_matches)} times.")
        for (x, y, w, h, conf) in filtered_matches:
            print(f" - Location: ({x}, {y}), Size: ({w}x{h}), Confidence: {conf:.2f}")
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{name} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print(f"Template '{name}' not found.")

# Display the final image with detections
cv2.imshow("Detected Templates", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
