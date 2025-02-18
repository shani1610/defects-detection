# defects-detection

File Structure:
kotlin

```
project_root/
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