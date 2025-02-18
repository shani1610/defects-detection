import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GMMPipeline:

    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.prob_map = None
        self.defect_mask = None

    def train(self):
        # --- Train a GMM on the Reference Image ---
        # Flatten the reference grayscale image to a list of pixel intensities.
        X = self.img2.reshape(-1, 1).astype(np.float32)

        # Fit a Gaussian Mixture Model (with 5 components, you can tune this number)
        gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
        gmm.fit(X)

        # --- Apply the GMM to the Inspected Image ---
        # Flatten the inspected image
        Y = self.img1.reshape(-1, 1).astype(np.float32)

        # Compute the log-likelihood for each pixel in the inspected image
        log_prob = gmm.score_samples(Y)  # Returns log probability per sample
        # Convert to probability (exponentiate the log-likelihood)
        prob = np.exp(log_prob)

        # Reshape the probability values to the inspected image's shape
        prob_map = prob.reshape(self.img1.shape)
        self.prob_map = prob_map
        return prob_map
    
    def threshold(self, percentile=20):
        # --- Segment Defects Based on the Probability Map ---
        # Here, we assume that low probability under the background model indicates a defect.
        # For example, we can threshold at the 10th percentile.
        threshold = np.percentile(self.prob_map, percentile)
        defect_mask = (self.prob_map < threshold).astype(np.uint8) * 255
        self.defect_mask = defect_mask
        return defect_mask

    def visualize(self):
        # Visualize the results
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.prob_map, cmap='jet')
        plt.title("Probability Map (Background Likelihood)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self.defect_mask, cmap='gray')
        plt.title("Defect Segmentation")
        plt.axis("off")

        plt.show()

        
