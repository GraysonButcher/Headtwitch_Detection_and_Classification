# HTR Analysis Tool - Setup Guide

## Quick Setup with Virtual Environment

### Step 1: Create Virtual Environment
```bash
# Navigate to the project folder
cd "C:\Users\grays\Dropbox\HDAC"

# Create virtual environment (choose one method)
python -m venv htr_env
# OR if you have multiple Python versions:
python3 -m venv htr_env
```

### Step 2: Activate Virtual Environment
```bash
# On Windows:
htr_env\Scripts\activate

# On macOS/Linux (if needed later):
source htr_env/bin/activate
```

You should see `(htr_env)` at the beginning of your command prompt when activated.

### Step 3: Install Dependencies
```bash
# Make sure you're in the virtual environment (you should see (htr_env))
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Test Installation
```bash
python test_app.py
```

### Step 5: Run the Application
```bash
python main.py
```

## Managing the Virtual Environment

### Deactivate Environment (when done working)
```bash
deactivate
```

### Reactivate Environment (next time you work on this)
```bash
# Navigate back to project folder
cd "C:\Users\grays\Dropbox\HDAC"
# Activate environment
htr_env\Scripts\activate
```

### Delete Environment (if you want to start fresh)
```bash
# Deactivate first
deactivate
# Remove the entire folder
rmdir /s htr_env
```

## Why Use Virtual Environment?

✅ **Isolated Dependencies**: Packages installed here won't affect your system Python
✅ **Version Control**: Each project can have different package versions
✅ **Clean Uninstall**: Just delete the `htr_env` folder to remove everything
✅ **Reproducible**: Other users can create identical environments

## Troubleshooting

### If `python` command not found:
Try `py` instead of `python`:
```bash
py -m venv htr_env
```

### If virtual environment creation fails:
Make sure you have Python 3.9+ installed:
```bash
python --version
```

### If pip install fails:
Try upgrading pip first:
```bash
python -m pip install --upgrade pip
```

## What Gets Installed

The main packages that will be installed in your virtual environment:
- **PySide6**: GUI framework (Qt for Python)
- **OpenCV**: Video processing
- **scikit-learn & XGBoost**: Machine learning
- **pandas & numpy**: Data processing
- **matplotlib**: Plotting
- **h5py**: Reading SLEAP H5 files

Total size: ~500MB in the virtual environment folder.