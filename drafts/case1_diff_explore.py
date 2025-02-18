
### here we crop the image ###

import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure the core module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure the processed folder exists
processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
os.makedirs(processed_dir, exist_ok=True)  # Create directory if it doesn't exist

# Define correct save paths
aligned_img_path = os.path.join(processed_dir, "aligned_image.tif")
reference_img_path = os.path.join(processed_dir, "reference_image.tif")
mask_img_path = os.path.join(processed_dir, "warped_mask.tif")

def crop_to_mask(image, mask, reference):
    """ Crop the image and reference image to the bounding box of the valid region in the mask. """
    coords = cv2.findNonZero(mask)  # Find nonzero pixels in the mask
    x, y, w, h = cv2.boundingRect(coords)  # Get bounding box

    # Crop both aligned image and reference using the same bounding box
    cropped_image = image[y:y+h, x:x+w]
    cropped_reference = reference[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    return cropped_image, cropped_reference, cropped_mask

# Load images
aligned_img = cv2.imread(aligned_img_path, cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
warped_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

# Ensure mask is binary (sometimes saving/loading converts it)
_, warped_mask = cv2.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)

# Crop using mask
aligned_cropped, reference_cropped, mask_cropped = crop_to_mask(aligned_img, warped_mask, reference_img)

# Save cropped images
cropped_aligned_path = os.path.join(processed_dir, "aligned_cropped.tif")
cropped_reference_path = os.path.join(processed_dir, "reference_cropped.tif")
cropped_mask_path = os.path.join(processed_dir, "warped_mask_cropped.tif")

cv2.imwrite(cropped_aligned_path, aligned_cropped)
cv2.imwrite(cropped_reference_path, reference_cropped)
cv2.imwrite(cropped_mask_path, mask_cropped)

print(f"Saved cropped aligned image to {cropped_aligned_path}")
print(f"Saved cropped reference image to {cropped_reference_path}")
print(f"Saved cropped mask to {cropped_mask_path}")

# Compute the absolute difference using the cropped images

diff_img = cv2.absdiff(aligned_cropped, reference_cropped)

# Apply the cropped mask
diff_img[mask_cropped == 0] = 0

# Visualize the refined difference image
plt.figure(figsize=(10, 5))
plt.imshow(diff_img, cmap='hot')
plt.title("Refined Difference Image (Cropped Correctly)")
plt.axis("off")
plt.colorbar()
plt.show()
'''
#---------------
from scipy.signal import correlate2d

def correlation_filter(diff_img, reference_img):
    """ Apply correlation filtering to remove common structures and retain anomalies. """
    # Compute 2D correlation
    correlation_map = correlate2d(diff_img, reference_img, mode='same', boundary='symm')

    # Normalize correlation map
    correlation_map = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold: Keep low-correlation areas (potential defects)
    threshold = np.percentile(correlation_map, 80)  # Adjust percentile if needed
    filtered_defects = np.where(correlation_map < threshold, diff_img, 0)

    return filtered_defects

# Apply correlation-based filtering
filtered_defects = correlation_filter(diff_img, reference_cropped)

# Visualize result
plt.figure(figsize=(10, 5))
plt.imshow(filtered_defects, cmap='hot')
plt.title("Defect Detection using Correlation Filtering")
plt.axis("off")
plt.colorbar()
plt.show()
#--------
def binary_xor_defects(diff_img, reference_img, threshold=50):
    """ Perform binary XOR to highlight new structures in the inspected image. """
    # Convert images to binary
    _, bin_diff = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
    _, bin_ref = cv2.threshold(reference_img, threshold, 255, cv2.THRESH_BINARY)

    # Perform XOR operation to detect new structures
    xor_defects = cv2.bitwise_xor(bin_diff, bin_ref)

    return xor_defects

# Apply XOR operation
filtered_defects = binary_xor_defects(diff_img, reference_cropped)

# Visualize result
plt.figure(figsize=(10, 5))
plt.imshow(filtered_defects, cmap='gray')
plt.title("Defect Detection using XOR Operation")
plt.axis("off")
plt.show()
#---------
def convolve_with_edge_kernel(image):
    """ Apply a convolutional edge-detection filter to highlight defects. """
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])  # Laplacian Kernel for Edge Detection
    filtered = cv2.filter2D(image, -1, kernel)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Apply edge convolution to both images
reference_edges = convolve_with_edge_kernel(reference_cropped)
diff_edges = convolve_with_edge_kernel(diff_img)

# Remove common edges
filtered_defects = cv2.absdiff(diff_edges, reference_edges)

# Visualize result
plt.figure(figsize=(10, 5))
plt.imshow(filtered_defects, cmap='hot')
plt.title("Defect Detection using Convolution-based Edge Filtering")
plt.axis("off")
plt.colorbar()
plt.show()

# ------------------filters--------------------------
def fft_high_pass_filter(img, size=30):
    """ Compute high-pass FFT filter for an image. """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Create a high-pass mask (keep high frequencies, remove low frequencies)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-size:crow+size, ccol-size:ccol+size] = 0  # Remove low frequencies

    # Apply mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # Normalize result to 0-255
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_filtered

def fft_log_subtraction(reference_img, diff_img, size=30):
    """ Compute log magnitude FFT subtraction to enhance defect areas. """
    
    # Compute FFT
    f_ref = np.fft.fft2(reference_img)
    f_diff = np.fft.fft2(diff_img)
    
    # Shift FFT
    fshift_ref = np.fft.fftshift(f_ref)
    fshift_diff = np.fft.fftshift(f_diff)

    # Take log magnitude
    log_ref = np.log(np.abs(fshift_ref) + 1e-8)
    log_diff = np.log(np.abs(fshift_diff) + 1e-8)

    # Subtract log magnitudes
    log_filtered = log_diff - log_ref

    # Inverse FFT
    f_ishift = np.fft.ifftshift(np.exp(log_filtered))
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    # Normalize and return
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_filtered

filtered_defects = fft_log_subtraction(reference_cropped, diff_img, size=30)

plt.figure(figsize=(10, 5))
plt.imshow(filtered_defects, cmap='hot')
plt.title("FFT Log Magnitude Subtraction - Enhanced Defects")
plt.axis("off")
plt.colorbar()
plt.show()
#----------------------------------------------------


'''
'''
# Compute mean and standard deviation of the difference image
mean_diff = np.mean(diff_img)
std_diff = np.std(diff_img)

# Thresholding: Keep only pixels that are outside ±2*std, im a bit concern about it cuz we might loose one pixel defect
lower_bound = mean_diff - 2 * std_diff
upper_bound = mean_diff + 2 * std_diff
filtered_diff_img = np.where((diff_img < lower_bound) | (diff_img > upper_bound), diff_img, 0)

# Visualize the thresholded difference image
plt.figure(figsize=(10, 5))
plt.imshow(filtered_diff_img, cmap='hot')
plt.title("Thresholded Difference Image (Outside ±2σ)")
plt.axis("off")
plt.colorbar()
plt.show()

# Step 1: Extract edges from the reference image
edges_reference = cv2.Canny(align.img2, 50, 150)

# Step 2: Extract edges from the difference image
edges_difference = cv2.Canny(diff_img, 50, 150)

# Step 3: Subtract reference edges from the difference edges
#filtered_edges = cv2.bitwise_xor(edges_difference, edges_reference)

# Step 4: Use filtered edges to refine the difference image
#edge_filtered_diff = cv2.bitwise_and(diff_img, diff_img, mask=filtered_edges)

# Visualize the edge-filtered difference image
plt.figure(figsize=(10, 5))
plt.imshow(edges_reference, cmap='hot')
plt.title("edges_reference")
plt.axis("off")
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(edges_difference, cmap='hot')
plt.title("Eedges_difference")
plt.axis("off")
plt.colorbar()
plt.show()

bilateralFilter_img1 = cv2.bilateralFilter(diff_img, 9, 75, 75)

plt.figure(figsize=(10, 5))
plt.imshow(bilateralFilter_img1, cmap='hot')
plt.title("bilateralFilter_img1")
plt.axis("off")
plt.colorbar()
plt.show()

img_back = high_pass_fft(diff_img)

plt.figure(figsize=(10, 5))
plt.imshow(img_back, cmap='hot')
plt.title("high_pass_fft")
plt.axis("off")
plt.colorbar()
plt.show()
'''