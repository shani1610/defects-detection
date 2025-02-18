import cv2
import numpy as np
import matplotlib.pyplot as plt

class Postprocess:

    def __init__(self, binary_map, max_defects=5, ar_thresh=(0.5, 2.0), circularity_thresh=0.3, solidity_thresh=0.85):
        self.binary_map = binary_map
        self.max_defects = max_defects
        self.ar_thresh = ar_thresh
        self.circularity_thresh = circularity_thresh
        self.solidity_thresh = solidity_thresh
        self.filtered_mask = None 

    def filter_defect_candidates(self, binary_map, max_defects=5):
        """Keep only the top defect candidates in the binary defect map."""
        
        # Find all connected components (contours)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Create an empty mask for filtered defects
        filtered_mask = np.zeros_like(binary_map)

        # Keep only the top `max_defects` largest defects
        for i, contour in enumerate(contours[:max_defects]):
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        self.filtered_mask = filtered_mask
        return filtered_mask

    def filter_defects_by_shape(self, binary_map, max_defects=5, ar_thresh=(0.5, 2.0), circularity_thresh=0.3, solidity_thresh=0.85):
        """ Keep only the top defect candidates based on shape properties. """
        
        # Find contours of connected components
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Create an empty mask for filtered defects
        filtered_mask = np.zeros_like(binary_map)
        
        kept_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Ignore very small noise
                continue

            # Compute bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Compute circularity (roundness)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # Compute solidity (compactness)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Apply shape filtering conditions
            if (ar_thresh[0] <= aspect_ratio <= ar_thresh[1]) and (circularity > circularity_thresh) and (solidity > solidity_thresh):
                kept_contours.append(contour)
            
            # Stop after selecting `max_defects`
            if len(kept_contours) >= max_defects:
                break

        # Draw the selected contours on the mask
        cv2.drawContours(filtered_mask, kept_contours, -1, 255, thickness=cv2.FILLED)
        self.filtered_mask = filtered_mask

        return filtered_mask

    def visualize(self):
        # Visualize result
        plt.figure(figsize=(6, 5))
        plt.imshow(self.filtered_mask, cmap='gray')
        plt.title("Filtered Defect Candidates")
        plt.axis("off")
        plt.show()