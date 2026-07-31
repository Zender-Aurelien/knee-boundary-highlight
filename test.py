import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.morphology import binary_erosion, disk,binary_dilation

# 1. get all the BMP images in folder
image_paths = sorted(glob.glob(r'mri\9606664\images\*.bmp'))

if not image_paths:
    print("No images found. Please check your file path.")
else:
    # 2. Load first image to set up our initial manual guess
    first_img = io.imread(image_paths[0], as_gray=True)
    
    current_mask = np.zeros(first_img.shape, dtype=np.int8)
    yy, xx = np.ogrid[:first_img.shape[0], :first_img.shape[1]]
    
    circle = (yy - 220)**2 + (xx - 220)**2 <= 25**2
    current_mask[circle] = 1

    # Extract the patient ID from the first image's folder structure
    first_path = image_paths[0]
    path_parts = os.path.normpath(first_path).split(os.sep)
    patient_id = path_parts[1] # Grabs the folder name immediately after 'mri'

    # Create an output directory inside this specific patient's folder
    output_dir = os.path.join("mri", patient_id, "output_tracked_masks")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Tracking results will be saved to: {output_dir}")

    # 4. Loop through every image in the folder
    for index, path in enumerate(image_paths):
        print(f"Processing slice {index + 1}/{len(image_paths)}: {path}")
        
        # Load current slice and convert to a float (0.0 to 1.0)
        img = io.imread(path, as_gray=True)
        img = img.astype(np.float32) / 255.0
        
        # increase contrast: Map the 2nd and 98th percentiles to 0 and 1
        # This turns faint bone walls into solid black barriers
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
        
        #Switch behavior based on the slice index
        # First half: Bone is generally growing
        # Second half: Bone is generally shrinking
        midpoint = len(image_paths) // 2
        
        if index < 10:
            # GROWING PHASE: Start safely inside, then expand outward
            current_mask = binary_erosion(current_mask, disk(2))
            current_balloon = 0.9
        elif index >=10 and index < 19:
            # SHRINKING PHASE: Start safely outside, then shrink-wrap inward
            current_mask = binary_dilation(current_mask, disk(2))
            current_balloon = -0.85
        else: #expand outward to compensate for changes at end
            current_mask = binary_erosion(current_mask, disk(2))
            current_balloon = 0.75
        
        # Create the edge map for the current slice
        gimage = inverse_gaussian_gradient(img, alpha=5000.0, sigma=2)
        
        # run morphological snake
        current_mask = morphological_geodesic_active_contour(
            gimage, 
            num_iter=12, 
            init_level_set=current_mask, 
            smoothing=3, 
            balloon=current_balloon
        )
        
        # 5. Optional: Visualize the tracking in real-time
        # (Comment this block out if you just want it to process silently in the background)
        plt.clf() # Clear the previous plot
        plt.imshow(img, cmap='gray')
        plt.contour(current_mask, [0.5], colors='b', linewidths=2)
        plt.title(f"Tracking Bone: Slice {index + 1}")

        original_filename = os.path.basename(path)
        save_name = original_filename.replace('.bmp', '_tracked.png')
        save_path = os.path.join(output_dir, save_name)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.pause(0.1) # Pause briefly to update the window

    print("Sequence complete! All images saved.")
    plt.show()