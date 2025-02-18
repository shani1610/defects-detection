import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

class Preprocess:
    def __init__(self, img1, img2):
        self.img1 = self.load_image(img1)
        self.img2 = self.load_image(img2)
    
    def load_image(self, img):
        """ Load image if a path is given, otherwise assume it's an OpenCV image array. """
        if isinstance(img, str):
            return cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise ValueError("Invalid input: Expected a file path or an OpenCV image array.")
    
    def histogram_matching(self):
        """ Adjust img1 to match the histogram of img2. """
        matched_img = match_histograms(self.img1, self.img2)
        return matched_img.astype(np.uint8)
    
    def local_contrast_normalization(self, kernel_size=15):
        """ Apply Local Contrast Normalization (LCN) to both images. """
        def apply_lcn(image):
            mean = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            std = cv2.GaussianBlur((image - mean) ** 2, (kernel_size, kernel_size), 0) ** 0.5
            normalized_img = (image - mean) / (std + 1e-8)
            normalized_img = cv2.normalize(normalized_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return normalized_img
        
        img1_lcn = apply_lcn(self.img1)
        img2_lcn = apply_lcn(self.img2)
        return img1_lcn, img2_lcn
    
    def bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """ Apply bilateral filtering to reduce noise while preserving edges. """
        filtered_img1 = cv2.bilateralFilter(self.img1, d, sigma_color, sigma_space)
        filtered_img2 = cv2.bilateralFilter(self.img2, d, sigma_color, sigma_space)
        return filtered_img1, filtered_img2
    
    def high_pass_fft(self):
        """ Apply high-pass filtering using FFT to emphasize edges and fine details. """
        f = np.fft.fft2(self.img1)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img1.shape
        crow, ccol = rows // 2 , cols // 2
        
        # Create a mask with high-pass filter
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
        
        # Apply mask and inverse FFT
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize result to 0-255 range
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img_back
    
    def visualize(self, method='histogram_matching'):
        """ Visualize original images and the preprocessed result. """
        if method == 'histogram_matching':
            processed_img = self.histogram_matching()
            method_name = "Histogram Matching"
        elif method == 'lcn':
            img1_lcn, img2_lcn = self.local_contrast_normalization()
            processed_img = img1_lcn
            method_name = "Local Contrast Normalization (LCN)"
        elif method == 'bilateral':
            processed_img1, processed_img2 = self.bilateral_filter()
            processed_img = processed_img1
            method_name = "Bilateral Filtering"
        elif method == 'high_pass_fft':
            processed_img = self.high_pass_fft()
            method_name = "High-Pass FFT Filtering"
        else:
            raise ValueError("Invalid method: Choose 'histogram_matching', 'lcn', 'bilateral', or 'high_pass_fft'")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(self.img1, cmap='gray')
        axes[0].set_title("Original Inspected Image")
        axes[0].axis("off")
        
        axes[1].imshow(self.img2, cmap='gray')
        axes[1].set_title("Reference Image")
        axes[1].axis("off")
        
        axes[2].imshow(processed_img, cmap='gray')
        axes[2].set_title(f"Preprocessed Image ({method_name})")
        axes[2].axis("off")
        
        plt.show()
        return processed_img
    
# Example Usage:
# preprocess = Preprocess("image1.jpg", "image2.jpg")
# matched_img = preprocess.histogram_matching()
# img1_lcn, img2_lcn = preprocess.local_contrast_normalization()
# bilateral_img = preprocess.bilateral_filter()
# high_pass_img = preprocess.high_pass_fft()
# preprocess.visualize(method='histogram_matching')
# preprocess.visualize(method='lcn')
# preprocess.visualize(method='bilateral')
# preprocess.visualize(method='high_pass_fft')
   
    
