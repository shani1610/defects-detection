# Defect Detection in Semiconductor Chips

This project focuses on detecting and analyzing defects in semiconductor chips by aligning inspected and reference images.


File Structure:

```
defect-detection/
│── core/
│   ├── __init__.py
│   ├── visualize.py
│── scripts/
│   ├── defective_case1.py
│── data/
│   ├── defective/
│       ├── case1_inspected_image.tif
│       ├── case1_reference_image.tif
```

# Installation:

Clone the repository 

```
git clone https://github.com/shani1610/defects-detection.git
cd defects-detection
```

Create a Virtual Environment

```
python -m venv venv
```

Activate the Virtual Environment

On Windows (Command Prompt):

```
venv\Scripts\activate
```

On macOS/Linux:

```
source venv/bin/activate
```

Install Required Packages

```
pip install -r requirements.txt
```

# Running:

python scripts/defective_case1.py