import cv2
import numpy as np
import matplotlib.pyplot as plt

class FFTPipeline:
    def __init__(self, img1, img2, homography, gaus_kernel, size):
        self.img1 = img1
        self.img2 = img2
        self.homography = homography
        self.gaus_kernel = gaus_kernel
        self.size = size
        self.filtered_defect_map = None

    def high_pass_filter(self, img, size):
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
        
        print(f"fshift shape: {fshift.shape} ")
        print(f"mask shape: {mask.shape} ")
        print(f"img shape: {img.shape} ")

        # Visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.log1p(np.abs(fshift)), cmap='gray')
        plt.title("FFT Magnitude Spectrum")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"High-Pass Filter Mask, size: {size}")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(np.log1p(np.abs(fshift_filtered)), cmap='gray')
        plt.title("Filtered FFT Spectrum")
        plt.axis("off")
        
        plt.show()

        return fshift_filtered  # Keep in frequency domain

    def inverse_fft(self, fshift_filtered):
        """ Convert filtered frequency domain image back to spatial domain. """
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        print(f"f_ishift shape: {f_ishift.shape} ")
        print(f"img_back shape: {img_back.shape} ")

        # Normalize result to 0-255 range
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img_back

    def applyftt(self):
        # Apply Gaussian Blur before Fourier Transform
        self.img2 = cv2.GaussianBlur(self.img2, (self.gaus_kernel, self.gaus_kernel), 0)
        self.img1 = cv2.GaussianBlur(self.img1, (self.gaus_kernel, self.gaus_kernel), 0)
        
        # Step 1: Move both images to frequency domain and apply high-pass filter
        fshift_ref = self.high_pass_filter(self.img2, self.size) # i choose 2 cuz bigger delete the defects
        fshift_inspected = self.high_pass_filter(self.img1, self.size)

        # Step 2: Compute the difference in frequency domain
        fshift_diff = fshift_inspected - fshift_ref
    
        # Step 3: Move back to spatial domain
        filtered_defect_map = self.inverse_fft(fshift_diff)

        h, w = self.img2.shape  # Reference image dimensions
        mask = np.ones_like(self.img1, dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, self.homography, (w, h))

        filtered_defect_map[warped_mask == 0] = 0
        self.filtered_defect_map = filtered_defect_map
        
        return filtered_defect_map
    
    def visualize(self):
        plt.subplot(1, 3, 3)
        plt.imshow(self.filtered_defect_map, cmap='hot')
        plt.title(f"Defect Map (FFT-Based High-Pass Filtering), Gauss Kernel = {self.gaus_kernel}, Filter Size = {self.size}")
        plt.axis("off")
        plt.colorbar()
        plt.show()

