#!/bin/bash

# Ensure the script stops if an error occurs
set -e

# Activate virtual environment if needed
source venv/bin/activate  # Use this if you have a virtual environment

# Run defect detection for different cases
echo "Running defect detection for Case 1..."
python main.py data/defective/case1_inspected_image.tif data/defective/case1_reference_image.tif

echo "Running defect detection for Case 2..."
python main.py data/defective/case2_inspected_image.tif data/defective/case2_reference_image.tif

echo "Running defect detection for Case 3 (Non-Defective)..."
python main.py data/non_defective/case3_inspected_image.tif data/non_defective/case3_reference_image.tif

echo "All cases completed successfully!"
