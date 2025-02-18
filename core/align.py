import cv2
import numpy as np
import matplotlib.pyplot as plt

class Align:
    def __init__(self, img1, img2, method='SIFT', refine=False, plot=True):
        self.img1 = self.load_image(img1)
        self.img2 = self.load_image(img2)
        self.method = method.upper()
        self.refine = refine
        self.plot = plot
        self.keypoints1, self.descriptors1 = None, None
        self.keypoints2, self.descriptors2 = None, None
        self.matches = None
        self.homography = None
        self.inliers = None
        self.outliers = None
        self.aligned_img = None
    
    def load_image(self, img):
        """ Load image if a path is given, otherwise assume it's an OpenCV image array. """
        if isinstance(img, str):  # If it's a file path, read the image
            return cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        elif isinstance(img, np.ndarray):  # If it's already an image array, use it directly
            return img
        else:
            raise ValueError("Invalid input: Expected a file path or an OpenCV image array.")

    
    def detect_keypoints(self):
        """ Detect keypoints and descriptors using SIFT or ORB. """
        if self.method == 'SIFT':
            detector = cv2.SIFT_create()
        elif self.method == 'ORB':
            detector = cv2.ORB_create()
        else:
            raise ValueError("Invalid method: Choose 'SIFT' or 'ORB'")
        
        self.keypoints1, self.descriptors1 = detector.detectAndCompute(self.img1, None)
        self.keypoints2, self.descriptors2 = detector.detectAndCompute(self.img2, None)
        
        if self.plot:
            self.visualize_keypoints()
    
    def visualize_keypoints(self):
        """ Draw and display detected keypoints on both images. """
        img1_kp = cv2.drawKeypoints(self.img1, self.keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_kp = cv2.drawKeypoints(self.img2, self.keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1_kp, cmap='gray')
        axes[0].set_title("Keypoints in Image 1")
        axes[0].axis("off")
        
        axes[1].imshow(img2_kp, cmap='gray')
        axes[1].set_title("Keypoints in Image 2")
        axes[1].axis("off")
        
        plt.show()
    
    def match_keypoints(self):
        """ Match keypoints using BFMatcher (for ORB) or FLANN (for SIFT). """
        if self.method == 'SIFT':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.matches = matcher.match(self.descriptors1, self.descriptors2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)  # Sort by quality
        
        if self.plot:
            self.visualize_matches()
    
    def visualize_matches(self):
        """ Draw and display matched keypoints. """
        img_matches = cv2.drawMatches(self.img1, self.keypoints1, self.img2, self.keypoints2, self.matches[:50], None, flags=2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches)
        plt.title(f"Top {len(self.matches[:50])} Matches ({self.method})")
        plt.axis("off")
        plt.show()
    
    def compute_homography(self):
        """ Compute homography using RANSAC and extract inliers & outliers. """
        src_pts = np.float32([self.keypoints1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.keypoints2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        
        self.homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.inliers = [self.matches[i] for i in range(len(self.matches)) if mask[i]]
        self.outliers = [self.matches[i] for i in range(len(self.matches)) if not mask[i]]
        
        if self.plot:
            self.visualize_inliers()

        return self.homography, self.inliers, self.outliers
    
    def visualize_inliers(self):
        """ Draw matches, distinguishing inliers and outliers from RANSAC. """
        img_inliers = cv2.drawMatches(self.img1, self.keypoints1, self.img2, self.keypoints2, self.inliers, None, flags=2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(img_inliers)
        plt.title(f"Inlier Matches ({len(self.inliers)} / {len(self.matches)})")
        plt.axis("off")
        plt.show()

    def refine_ecc(self):    
        warp_matrix = self.homography.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)
        try:
            cc, refined_warp_matrix = cv2.findTransformECC(self.img2, self.img1, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
        except cv2.error as e:
            print("ECC refinement error: ", e)
        return refined_warp_matrix
    
    def align_images(self):
        """ Apply the homography to align img1 to img2 and visualize the overlay. """
        if self.homography is None:
            raise ValueError("Homography not computed. Run compute_homography() first.")
        
        h, w = self.img2.shape
        self.aligned_img = cv2.warpPerspective(self.img1, self.homography, (w, h))
        
        if self.plot:
            plt.figure(figsize=(10, 5))
            plt.imshow(self.img2, cmap='gray', alpha=0.5, label='Reference Image')
            plt.imshow(self.aligned_img, cmap='jet', alpha=0.5, label='Aligned Image')
            plt.title("Overlay of Aligned Image on Reference Image")
            plt.axis("off")
            plt.show()

        if self.refine: # TO DO: fix and check if the projection error in lower
            refined_warp_matrix = self.refine_ecc()
            self.aligned_img = cv2.warpPerspective(self.img1, self.homography, (w, h))
            
            if self.plot:
                plt.figure(figsize=(10, 5))
                plt.imshow(self.img2, cmap='gray', alpha=0.5, label='Reference Image')
                plt.imshow(self.aligned_img, cmap='jet', alpha=0.5, label='Aligned Image')
                plt.title("Overlay of Aligned Image on Reference Image (ECC Refined)")
                plt.axis("off")
                plt.show()
            
    def compute_reprojection_error(self):
        """ Compute the reprojection error for the estimated homography. """
        if self.homography is None:
            raise ValueError("Homography not computed. Run compute_homography() first.")
        
        src_pts = np.float32([self.keypoints1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.keypoints2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        
        projected_pts = cv2.perspectiveTransform(src_pts, self.homography)
        error = np.mean(np.linalg.norm(projected_pts - dst_pts, axis=2))
        
        print(f"Reprojection Error: {error:.4f}")
        return error

# Example Usage:
# align = Align("image1.jpg", "image2.jpg", method='SIFT')
# align.detect_keypoints()
# align.match_keypoints()
# homography, inliers, outliers = align.compute_homography()
# align.align_images()
# # error = align.compute_reprojection_error()
