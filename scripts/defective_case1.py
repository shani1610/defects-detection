import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# Ensure the core module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.visualize import Visualize
from core.analyze import Analyze  # Import the new Analyze class
from core.align import Align  # Import the new Analyze class
from core.preprocess import Preprocess  # Import the new Analyze class
from core.gmmpipe import GMMPipeline  # Import the new Analyze class
from core.fftpipe import FFTPipeline  # Import the new Analyze class
from core.postprocess import Postprocess  # Import the new Analyze class

plot = True

# Paths to images
image1_path = "data/defective/case2_inspected_image.tif"
image2_path = "data/defective/case2_reference_image.tif"
defects_file = "data/defective/defects locations.txt"

# Create visualization instance and display images
vis = Visualize(image1_path, image2_path)
if plot:
    vis.show_two_images()
    #vis.visualize_defects(defects_file)
    #vis.segment_defects()

# Perform analysis on the images
analyze = Analyze(image1_path, image2_path)
if plot:
    analyze.visualize()  # This will display images, histograms, and stats
comparison = analyze.compare_histograms(analyze.images[0], analyze.images[1])
print(comparison)

# Image Alignment 
align = Align(image1_path, image2_path, method='ORB', refine=False, plot=plot)
align.detect_keypoints()
align.match_keypoints()
homography, inliers, outliers = align.compute_homography() #outliers didnt indicate anything useful 
align.align_images()
error = align.compute_reprojection_error()

# Create a mask for the valid region in img1, Compute the difference image while excluding non-overlapping areas
h, w = align.img2.shape  # Reference image dimensions
mask = np.ones_like(align.img1, dtype=np.uint8) * 255
warped_mask = cv2.warpPerspective(mask, align.homography, (w, h))
diff_img = cv2.absdiff(align.aligned_img, align.img2)
diff_img[warped_mask == 0] = 0 # Apply the warped mask to remove non-overlapping areas
analyze_diff = Analyze(diff_img)
if plot:
    analyze_diff.visualize()

# ------------------- Producting Defect Map -----------------------------------------
# --- 1: GMM ------------------------------------------------------------------------
n_components=10
percentile=20
gmmpipe = GMMPipeline(align.aligned_img, align.img2, n_components=n_components, percentile=percentile)
prob_map = gmmpipe.train()
defect_mask = gmmpipe.threshold()
gmmpipe.visualize()
# Normalize the GMM probability map to [0, 1]
# Here, we assume lower probability indicates a defect, so invert it:
normalized_prob = cv2.normalize(prob_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
confidence_gmm = 1.0 - normalized_prob  # higher values indicate higher defect likelihood

# --- 2: FFT ------------------------------------------------------------------------
gaus_kernel = 13
size = 48
high_pass_filter = False
fftpipe = FFTPipeline(align.aligned_img, align.img2, high_pass_filter, align.homography, gaus_kernel = gaus_kernel, size = size)
filtered_defect_map = fftpipe.applyftt()
fftpipe.visualize()
# Calculate lower and upper percentile thresholds
lower = np.percentile(filtered_defect_map, 1)
upper = np.percentile(filtered_defect_map, 99)
# Clip the defect map
clipped = np.clip(filtered_defect_map, lower, upper)
# Normalize the clipped values to [0, 1]
confidence_fft = cv2.normalize(clipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# ----------------- Voting Mechanism -----------------------------------------------
# Multiply the two confidence maps
combined_confidence = confidence_gmm * confidence_fft

# Optionally, visualize the confidence maps before thresholding
plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
plt.imshow(confidence_gmm, cmap='jet')
plt.title(f"GMM Defect Confidence\n num components = {n_components}\n percentile = {percentile}", fontsize=12, fontweight='bold')
plt.colorbar()
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(confidence_fft, cmap='jet')
plt.title(f"FFT Defect Confidence\n Gaus Kernel = {gaus_kernel}\n size = {size}", fontsize=12, fontweight='bold')
plt.axis("off")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(combined_confidence, cmap='jet')
plt.title("Combined Confidence\n (multiplicaiton)", fontsize=12, fontweight='bold')
plt.axis("off")
plt.colorbar()
plt.show()

# Now threshold the combined confidence to get a binary defect mask.
# The threshold value (e.g., 0.5) might need to be adjusted.
threshold_val = 0.5
_, final_voted_mask = cv2.threshold(combined_confidence, threshold_val, 1.0, cv2.THRESH_BINARY)
final_voted_mask = (final_voted_mask * 255).astype(np.uint8)

# Visualize the final binary mask
plt.figure(figsize=(6, 6))
plt.imshow(final_voted_mask, cmap='gray')
plt.title(f"Final Voted Defect Mask\n threshold val = {threshold_val}")
plt.axis("off")
plt.show()

# ---------- Postprocessing ------------------------------------------
postprocess = Postprocess(final_voted_mask, max_defects=5, ar_thresh=(0.5, 2.0), circularity_thresh=0.3, solidity_thresh=0.85)
# Filter and retain only object candidates
filtered_mask = postprocess.filter_defect_candidates(final_voted_mask, max_defects=5)
postprocess.visualize()
# Filter defects based on shape
#filtered_mask = postprocess.filter_defects_by_shape(final_voted_mask, max_defects=5, ar_thresh=(0.5, 2.0), circularity_thresh=0.3, solidity_thresh=0.85)
#postprocess.visualize()

