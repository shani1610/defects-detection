import sys
import os
import argparse
import cv2
import numpy as np

# Ensure the core module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.visualize import Visualize
from core.analyze import Analyze  
from core.align import Align  
from core.preprocess import Preprocess 
from core.gmmpipe import GMMPipeline  
from core.fftpipe import FFTPipeline  
from core.postprocess import Postprocess  

def main(image1_path, image2_path, defects_file=None, plot=False):
    """ Main function to run the defect detection pipeline. """
    # ------------------- Examine ------------------------------------------------------
    vis = Visualize(image1_path, image2_path)
    if plot:
        vis.show_two_images()
        if defects_file:
            vis.visualize_defects(defects_file)
        vis.segment_defects()
    analyze = Analyze(image1_path, image2_path)
    if plot:
        analyze.visualize()  # Display images, histograms, and stats
    comparison = analyze.compare_histograms(analyze.images[0], analyze.images[1])
    #print(comparison)

    # ------------------------ Image Alignment ------------------------------------------
    align = Align(image1_path, image2_path, method='ORB', refine=False, plot=plot)
    align.detect_keypoints()
    align.match_keypoints()
    homography, inliers, outliers = align.compute_homography()
    align.align_images()
    #error = align.compute_reprojection_error()

    # ------------------- Difference Image ----------------------------------------------
    h, w = align.img2.shape
    mask = np.ones_like(align.img1, dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, align.homography, (w, h))
    diff_img = cv2.absdiff(align.aligned_img, align.img2)
    diff_img[warped_mask == 0] = 0  # Apply the warped mask to remove non-overlapping areas

    analyze_diff = Analyze(diff_img)
    if plot:
        analyze_diff.visualize()

    # ------------------- Producing Defect Map ------------------------------------------
    # --- 1: GMM ------------------------------------------------------------------------
    n_components = 10
    percentile = 20
    gmmpipe = GMMPipeline(align.aligned_img, align.img2, n_components=n_components, percentile=percentile)
    prob_map = gmmpipe.train()
    defect_mask = gmmpipe.threshold()
    if plot:
        gmmpipe.visualize()
    normalized_prob = cv2.normalize(prob_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    confidence_gmm = 1.0 - normalized_prob  # Higher values indicate higher defect likelihood

    # --- 2: FFT -------------------------------------------------------------------------
    gaus_kernel = 13
    size = 48
    high_pass_filter = False
    fftpipe = FFTPipeline(align.aligned_img, align.img2, high_pass_filter, align.homography, gaus_kernel=gaus_kernel, size=size)
    filtered_defect_map = fftpipe.applyftt()
    if plot:
        fftpipe.visualize_freq_domain()
        fftpipe.visualize()
    lower = np.percentile(filtered_defect_map, 1)
    upper = np.percentile(filtered_defect_map, 99)
    clipped = np.clip(filtered_defect_map, lower, upper)
    confidence_fft = cv2.normalize(clipped, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # ----------------- Voting Mechanism --------------------------------------------------
    combined_confidence = confidence_gmm * confidence_fft  # TODO: try other operations such as average
    if plot:
        vis.plot_voting(confidence_gmm, confidence_fft, combined_confidence, n_components, percentile, gaus_kernel, size)
    threshold_val = 0.5
    _, final_voted_mask = cv2.threshold(combined_confidence, threshold_val, 1.0, cv2.THRESH_BINARY)
    final_voted_mask = (final_voted_mask * 255).astype(np.uint8)
    vis.final_voting(final_voted_mask, threshold_val)

    # ---------- Postprocessing ------------------------------------------------------------
    postprocess = Postprocess(final_voted_mask, max_defects=5, ar_thresh=(0.5, 2.0), circularity_thresh=0.3, solidity_thresh=0.85)
    filtered_mask = postprocess.filter_defect_candidates(final_voted_mask, max_defects=5)
    if plot:
        postprocess.visualize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the defect detection pipeline.")
    parser.add_argument("image1_path", type=str, help="Path to the inspected image")
    parser.add_argument("image2_path", type=str, help="Path to the reference image")
    parser.add_argument("--defects_file", type=str, default=None, help="Optional: Path to defects location file")
    parser.add_argument("--plot", action="store_true", help="Enable visualization")

    args = parser.parse_args()
    
    main(args.image1_path, args.image2_path, args.defects_file, args.plot)

