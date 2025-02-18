import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import tifffile as tiff
import os

class Analyze:
    def __init__(self, *images):
        self.images = [self.load_image(img) for img in images]
        self.image_paths = [img if isinstance(img, str) else None for img in images]
    
    def load_image(self, img):
        """
        Load an image if the input is a path, otherwise return the input assuming it's already an image array.
        """
        if isinstance(img, str):
            if img.lower().endswith(".tif") or img.lower().endswith(".tiff"):
                return tiff.imread(img)
            else:
                return cv2.imread(img, cv2.IMREAD_UNCHANGED)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise ValueError("Invalid input: Expected a file path or an OpenCV image array.")
    
    def compute_statistics(self, img):
        """
        Compute mean, variance, standard deviation, and entropy for the image.
        """
        img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(img_gray)
        variance = np.var(img_gray)
        stddev = np.std(img_gray)
        hist = np.histogram(img_gray, bins=256, range=(0, 256))[0]
        ent = entropy(hist, base=2)
        return {"Mean": mean, "Variance": variance, "StdDev": stddev, "Entropy": ent}
    
    def compute_histogram(self, img):
        """
        Compute histogram for the given image.
        """
        if len(img.shape) == 2:
            return cv2.calcHist([img], [0], None, [256], [0, 256])
        else:
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
            return hist_r, hist_g, hist_b
    
    def compare_histograms(self, img1, img2):
        """
        Compare histograms of two images using Bhattacharyya Distance and Chi-Square.
        """
        hist1 = self.compute_histogram(img1)
        hist2 = self.compute_histogram(img2)
        
        if isinstance(hist1, tuple):  # Color images
            bhattacharyya = np.mean([cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) for h1, h2 in zip(hist1, hist2)])
            chi_square = np.mean([cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR) for h1, h2 in zip(hist1, hist2)])
        else:  # Grayscale images
            bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
        return {"Bhattacharyya Distance": bhattacharyya, "Chi-Square": chi_square}
    
    def visualize(self):
        """
        Visualize images with their histograms and statistics.
        """
        num_images = len(self.images)
        fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
        
        if num_images == 1:
            axes = [axes]
        
        for i, img in enumerate(self.images):
            img_gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            stats = self.compute_statistics(img)
            
            # Show Image
            axes[i][0].imshow(img_gray, cmap='gray' if len(img.shape) == 2 else None)
            title = os.path.basename(self.image_paths[i]) if self.image_paths[i] else f"Image {i+1}"
            axes[i][0].set_title(title)
            axes[i][0].axis("off")
            
            # Show Histogram
            hist_data = self.compute_histogram(img)
            axes[i][1].set_title("Histogram")
            
            if len(img.shape) == 2:
                axes[i][1].plot(hist_data, color='black')
            else:
                axes[i][1].plot(hist_data[0], color='red', label='Red')
                axes[i][1].plot(hist_data[1], color='green', label='Green')
                axes[i][1].plot(hist_data[2], color='blue', label='Blue')
                axes[i][1].legend()
            
            # Display Statistics
            stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
            axes[i][1].text(0.95, 0.95, stats_text, transform=axes[i][1].transAxes, fontsize=10,
                             verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

# Example Usage:
# analyze = Analyze("image1.tif", "image2.tif")
# analyze.visualize()
# comparison = analyze.compare_histograms(analyze.images[0], analyze.images[1])
# print(comparison)
#
# or with OpenCV loaded images:
# img1 = cv2.imread("image1.tif")
# img2 = cv2.imread("image2.tif")
# analyze = Analyze(img1, img2)
# analyze.visualize()
# comparison = analyze.compare_histograms(img1, img2)
# print(comparison)
