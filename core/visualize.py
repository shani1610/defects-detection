import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import imageio
import os
from skimage.segmentation import flood

class Visualize:
    def __init__(self, img1, img2, img3=None):
        self.img1_path = img1 if isinstance(img1, str) else None
        self.img2_path = img2 if isinstance(img2, str) else None
        self.img3_path = img3 if isinstance(img3, str) else None
        
        self.img1 = self.load_image(img1)
        self.img2 = self.load_image(img2)
        self.img3 = self.load_image(img3) if img3 is not None else None
    
    def load_image(self, img):
        """
        Load an image if the input is a path, otherwise return the input assuming it's already an image array.
        Supports .tif files using tifffile or imageio.
        """
        if isinstance(img, str):  # Check if the input is a file path
            if img.lower().endswith(".tif") or img.lower().endswith(".tiff"):
                return tiff.imread(img)  # Use tifffile to read TIFF images
            else:
                return cv2.imread(img, cv2.IMREAD_UNCHANGED)  # Load other image formats
        elif isinstance(img, np.ndarray):  # Check if the input is an image array
            return img
        elif img is None:
            return None
        else:
            raise ValueError("Invalid input: Expected a file path or an OpenCV image array.")
    
    def extract_label(self, path, default):
        """
        Extracts label from filename if it contains 'reference' or 'inspected'.
        """
        if path:
            filename = os.path.basename(path).lower()
            if "reference" in filename:
                return "Reference"
            elif "inspected" in filename:
                return "Inspected"
        return default
    
    def extract_case_number(self, path):
        """Extracts case number from the filename."""
        if path:
            filename = os.path.basename(path).lower()
            case_number = ''.join(filter(str.isdigit, filename.split('_')[0]))  # Extracts number after 'case'
            return int(case_number) if case_number else None
        return None
    
    def get_defect_locations(self, defects_file, case_number):
        """Extract defect locations for a given case number."""
        defect_locations = []
        with open(defects_file, 'r') as file:
            lines = file.readlines()
        
        case_str = f"case {case_number}:"
        found_case = False
        
        for line in lines:
            line = line.strip().lower()
            if line.startswith(case_str):
                found_case = True
                continue
            
            if found_case:
                if line.startswith("case "):
                    break  # Stop reading when the next case starts
                
                if "defect" in line and "x=" in line and "y=" in line:
                    parts = line.split('x=')[1]
                    x, y = map(int, parts.replace('y=', '').split(','))
                    defect_locations.append((x, y))
        
        return defect_locations
    
    def visualize_defects(self, defects_file):
        """Visualizes defects on the inspected image by overlaying circles at defect locations."""
        if self.img1_path:
            case_number = self.extract_case_number(self.img1_path)
            if case_number is None:
                print("Could not determine case number from filename.")
                return
            
            defect_locations = self.get_defect_locations(defects_file, case_number)
            if not defect_locations:
                print(f"No defect locations found for case {case_number}.")
                return
            
            img_copy = self.img1.copy()
            for i, (x, y) in enumerate(defect_locations, 1):
                cv2.circle(img_copy, (x, y), 10, (0, 0, 255), 2)  # Draw circle
                cv2.putText(img_copy, f"#{i}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img_copy, cmap='gray')
            plt.title(f"Defects Visualization - Case {case_number}")
            plt.axis("off")
            plt.show()
        else:
            print("No inspected image found to visualize defects.")

    def segment_defects(self, centers, tolerance=10):
        """
        Segments defects in the inspected image using a flood-fill algorithm from given center locations.
        
        Parameters:
        - centers: List of tuples (x, y) representing the defect centers.
        - tolerance: Intensity tolerance for region growing.
        
        Returns:
        - mask: A binary mask (boolean NumPy array) where True indicates the defect regions.
        """
        if self.img1 is None:
            print("No inspected image loaded for segmentation.")
            return None
        
        # Convert image to grayscale if necessary
        if len(self.img1.shape) > 2:
            img_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = self.img1
        
        # Initialize an empty mask
        mask = np.zeros(img_gray.shape, dtype=bool)
        
        for center in centers:
            # skimage's flood expects (row, col) i.e. (y, x)
            seed_point = (center[1], center[0])
            region_mask = flood(img_gray, seed_point=seed_point, tolerance=tolerance)
            mask = np.logical_or(mask, region_mask)

        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.axis("off")
        plt.show()

        return mask
    
    def show_two_images(self):
        """
        Display the two images side by side using Matplotlib.
        """
        if self.img1 is None or self.img2 is None:
            raise ValueError("One or both images could not be loaded.")
        
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB) if len(self.img1.shape) == 3 else self.img1
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB) if len(self.img2.shape) == 3 else self.img2
        
        label1 = self.extract_label(self.img1_path, "Image 1")
        label2 = self.extract_label(self.img2_path, "Image 2")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1_rgb, cmap='gray' if len(self.img1.shape) == 2 else None)
        axes[0].set_title(label1)
        axes[0].axis("off")
        
        axes[1].imshow(img2_rgb, cmap='gray' if len(self.img2.shape) == 2 else None)
        axes[1].set_title(label2)
        axes[1].axis("off")
        
        plt.show()
    
    def show_three_images(self):
        """
        Display three images side by side using Matplotlib.
        """
        if self.img1 is None or self.img2 is None or self.img3 is None:
            raise ValueError("One or more images could not be loaded.")
        
        img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB) if len(self.img1.shape) == 3 else self.img1
        img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB) if len(self.img2.shape) == 3 else self.img2
        img3_rgb = cv2.cvtColor(self.img3, cv2.COLOR_BGR2RGB) if len(self.img3.shape) == 3 else self.img3
        
        label1 = self.extract_label(self.img1_path, "Image 1")
        label2 = self.extract_label(self.img2_path, "Image 2")
        label3 = self.extract_label(self.img3_path, "Image 3")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img1_rgb, cmap='gray' if len(self.img1.shape) == 2 else None)
        axes[0].set_title(label1)
        axes[0].axis("off")
        
        axes[1].imshow(img2_rgb, cmap='gray' if len(self.img2.shape) == 2 else None)
        axes[1].set_title(label2)
        axes[1].axis("off")
        
        axes[2].imshow(img3_rgb, cmap='gray' if len(self.img3.shape) == 2 else None)
        axes[2].set_title(label3)
        axes[2].axis("off")
        
        plt.show()

# Example Usage:
# vis = Visualize("image1.jpg", "image2.jpg")
# vis.show_two_images()
#
# vis = Visualize("reference.jpg", "inspected.jpg")
# vis.show_two_images()
#
# vis = Visualize("image1.jpg", "image2.jpg", "image3.tif")
# vis.show_three_images()
#
# or with OpenCV loaded images:
# img1 = cv2.imread("image1.jpg")
# img2 = cv2.imread("image2.jpg")
# img3 = tiff.imread("image3.tif")
# vis = Visualize(img1, img2, img3)
# vis.show_three_images()
