import cv2
import numpy as np

def filter_islands_and_highlight(image_path, min_area=500):
    # 1. Load the original MRI image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 2. Apply a median blur to reduce speckle noise while preserving edges
    blurred = cv2.medianBlur(image, 5)

    # 3. Threshold the image to create a binary mask
    # Otsu's method automatically calculates the optimal threshold value
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Perform Connected Component Analysis
    # stats contains [x, y, width, height, area] for each label
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create a blank mask to store the filtered results
    filtered_mask = np.zeros_like(binary_mask)

    # 5. Island Filtering
    # Loop through all found components. 
    # We start at 1 to skip label 0, which is the background.
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # If the component's area is larger than our threshold, keep it
        if area >= min_area:
            filtered_mask[labels == i] = 255

    # 6. Extract the boundaries (contours) from the cleaned-up mask
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the original grayscale image to BGR so we can draw colored highlights
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the boundaries in red with a thickness of 2
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)

    return output_image, filtered_mask, contours

# --- Execution ---
image_path = "mri\9606664\images\9606664_07.bmp"
highlighted_result, clean_mask, bone_boundaries = filter_islands_and_highlight(image_path, min_area=800)

# Display the results
cv2.imshow("Original MRI", cv2.imread(image_path))
cv2.imshow("Filtered Bone Mask", clean_mask)
cv2.imshow("Highlighted Boundaries", highlighted_result)

cv2.waitKey(0)
cv2.destroyAllWindows()