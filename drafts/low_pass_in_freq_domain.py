import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_edge_map(img):
    """ Generate an edge map using the Canny edge detector. """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 20, 200)  # Detect edges (tune thresholds if needed)
    return edges


# Ensure the core module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure the processed folder exists
processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
os.makedirs(processed_dir, exist_ok=True)  # Create directory if it doesn't exist

def high_pass_filter(img, size=30):
    """ Apply high-pass filtering using FFT to retain only high-frequency details. """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # Get image dimensions
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    # Create high-pass filter mask
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-size:crow+size, ccol-size:ccol+size] = 0  # Block low frequencies

    # Apply mask and inverse FFT
    fshift_filtered = fshift * mask
    return fshift_filtered  # Keep in frequency domain

def inverse_fft(fshift_filtered):
    """ Convert filtered frequency domain image back to spatial domain. """
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize result to 0-255 range
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

def local_correlation(img1, img2, window_size=5):
    """ Compute local correlation instead of full-image correlation. """
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    
    # Compute local means
    mean1 = cv2.filter2D(img1, -1, kernel)
    mean2 = cv2.filter2D(img2, -1, kernel)
    
    # Compute local variances and covariance
    var1 = cv2.filter2D(img1**2, -1, kernel) - mean1**2
    var2 = cv2.filter2D(img2**2, -1, kernel) - mean2**2
    cov12 = cv2.filter2D(img1 * img2, -1, kernel) - mean1 * mean2
    
    # Compute correlation coefficient (avoid division by zero)
    correlation = cov12 / (np.sqrt(var1) * np.sqrt(var2) + 1e-8)
    
    return correlation

def phase_correlation_alignment(img1, img2):
    """ Perform subpixel alignment using phase correlation. """
    img1_float = np.float32(img1)
    img2_float = np.float32(img2)

    shift = cv2.phaseCorrelate(img1_float, img2_float)[0]
    dx, dy = int(shift[0]), int(shift[1])

    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_img = cv2.warpAffine(img2, translation_matrix, (img2.shape[1], img2.shape[0]))

    return aligned_img


# Load images
reference_img = cv2.imread("data/processed/reference_cropped.tif", cv2.IMREAD_GRAYSCALE)
inspected_img = cv2.imread("data/processed/aligned_cropped.tif", cv2.IMREAD_GRAYSCALE)

# Apply phase correlation alignment
#aligned_refined = phase_correlation_alignment(inspected_imgreference_img, )


# Apply edge detection to the filtered defect map
# Apply Gaussian Blur before Fourier Transform
reference_img = cv2.GaussianBlur(reference_img, (5, 5), 0)
inspected_img = cv2.GaussianBlur(inspected_img, (5, 5), 0)

edge_mapreference_img = create_edge_map(reference_img)

# Visualize the edge map
plt.figure(figsize=(6, 5))
plt.imshow(edge_mapreference_img, cmap='gray')
plt.title("Edge Map (Canny Edge Detector)")
plt.axis("off")
plt.show()
edge_maprinspected_img = create_edge_map(inspected_img)

# Visualize the edge map
plt.figure(figsize=(6, 5))
plt.imshow(edge_maprinspected_img, cmap='gray')
plt.title("Edge Map (Canny Edge Detector)")
plt.axis("off")
plt.show()
edge_diff = abs(edge_maprinspected_img - edge_mapreference_img)
plt.figure(figsize=(6, 5))
plt.imshow(edge_diff, cmap='gray')
plt.title("Edge Map (Canny Edge Detector)")
plt.axis("off")
plt.show()
def histogram_equalization(img):
    """ Apply Histogram Equalization to enhance contrast. """
    return cv2.equalizeHist(img)

# Apply Histogram Equalization to both images
enhanced_ref = histogram_equalization(reference_img)
enhanced_inspected = histogram_equalization(inspected_img)

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(enhanced_ref, cmap='gray')
plt.title("Reference Image (Enhanced)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_inspected, cmap='gray')
plt.title("Inspected Image (Enhanced)")
plt.axis("off")

plt.show()

# Step 4: Compute **localized** normalized cross-correlation
corr_map = local_correlation(inspected_img.astype(np.float32), reference_img.astype(np.float32))

# Normalize correlation for better visualization
corr_map = cv2.normalize(corr_map, None, 0, 1, cv2.NORM_MINMAX)

# Visualize result
plt.figure(figsize=(6, 5))
plt.imshow(corr_map, cmap='hot')
plt.title("Localized Normalized Cross-Correlation Map")
plt.axis("off")
plt.colorbar()
plt.show()

# Step 5: Invert correlation map (high correlation → 0, low correlation → 1)
corr_mask = 1 - corr_map  # Areas with high correlation become 0, low correlation become 1

# Apply mask to the difference image
inspected_img1 = inspected_img * corr_mask

# Normalize for better visualization
inspected_img1 = cv2.normalize(inspected_img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Visualize the final filtered defect map
plt.figure(figsize=(6, 5))
plt.imshow(inspected_img1, cmap='hot')
plt.title("inspected_img1 (Masked by Correlation)")
plt.axis("off")
plt.colorbar()
plt.show()

# Apply Gaussian Blur before Fourier Transform
reference_img = cv2.GaussianBlur(reference_img, (13, 13), 0)
inspected_img = cv2.GaussianBlur(inspected_img, (13, 13), 0)

# Step 1: Move both images to frequency domain and apply high-pass filter
fshift_ref = high_pass_filter(reference_img, size=2) # i choose 2 cuz bigger delete the defects
fshift_inspected = high_pass_filter(inspected_img, size=2)

# Step 2: Compute the difference in frequency domain
fshift_diff = fshift_inspected - fshift_ref

# Step 3: Move back to spatial domain
filtered_defect_map = inverse_fft(fshift_diff)

# Visualize the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(reference_img, cmap='gray')
plt.title("Reference Image (Original, Gaussian Blurred)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(inspected_img, cmap='gray')
plt.title("Inspected Image (Original, Gaussian Blurred)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(filtered_defect_map, cmap='hot')
plt.title("Defect Map (FFT-Based High-Pass Filtering)")
plt.axis("off")
plt.colorbar()

plt.show()



# Step 4: Compute **localized** normalized cross-correlation
corr_map = local_correlation(filtered_defect_map.astype(np.float32), reference_img.astype(np.float32))

# Normalize correlation for better visualization
corr_map = cv2.normalize(corr_map, None, 0, 1, cv2.NORM_MINMAX)

# Visualize result
plt.figure(figsize=(6, 5))
plt.imshow(corr_map, cmap='hot')
plt.title("Localized Normalized Cross-Correlation Map")
plt.axis("off")
plt.colorbar()
plt.show()

# Step 5: Invert correlation map (high correlation → 0, low correlation → 1)
corr_mask = 1 - corr_map  # Areas with high correlation become 0, low correlation become 1

# Apply mask to the difference image
filtered_diff = filtered_defect_map * corr_mask

# Normalize for better visualization
filtered_diff = cv2.normalize(filtered_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Visualize the final filtered defect map
plt.figure(figsize=(6, 5))
plt.imshow(filtered_diff, cmap='hot')
plt.title("Final Defect Map (Masked by Correlation)")
plt.axis("off")
plt.colorbar()
plt.show()






'''
# ------------------- Postprocessing -----------------------------------------
# Define a structuring element (kernel). You can experiment with kernel size.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# First, erode the final_mask to remove isolated noise.
eroded_mask = cv2.erode(final_voted_mask, kernel, iterations=1)
# Then, apply a closing (dilation followed by erosion) to fill in small gaps.
clean_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Compute Edge Map of the Reference Image ---
# Adjust thresholds as needed for your images
blurred_img2 = cv2.GaussianBlur(align.img2, (5, 5), 0)
edges = cv2.Canny(blurred_img2, threshold1=50, threshold2=150)

# Optionally, dilate the edges to cover a broader area
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)

# Invert the edge map so that the edges are 0 and non-edges are 255
inverted_edges = cv2.bitwise_not(dilated_edges)

# --- Remove Edge Regions from the Segmentation ---
# This will force the areas where the reference has strong edges to be background.
final_mask = cv2.bitwise_and(clean_mask, clean_mask, mask=inverted_edges)

# --- Visualization ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Reference image in color and grayscale
axs[0, 0].imshow(eroded_mask, cmap='gray')
axs[0, 0].set_title("eroded_mask")
axs[0, 0].axis("off")

# The probability map (use a colormap to see variations)
im = axs[0, 1].imshow(clean_mask, cmap='gray')
axs[0, 1].set_title("clean_mask")
axs[0, 1].axis("off")

axs[0, 2].imshow(edges, cmap='gray')
axs[0, 2].set_title("Edge Map of reference image (Canny)")
axs[0, 2].axis("off")

axs[1, 0].imshow(dilated_edges, cmap='gray')
axs[1, 0].set_title("dilated_edges of reference image (Canny)")
axs[1, 0].axis("off")

axs[1, 1].imshow(inverted_edges, cmap='gray')
axs[1, 1].set_title("inverted_edges of reference image (Canny)")
axs[1, 1].axis("off")

# Final Defect Mask after Removing Edge Regions
axs[1, 2].imshow(final_mask, cmap='gray')
axs[1, 2].set_title("Final Mask (Edges Removed)")
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()
'''