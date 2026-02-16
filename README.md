<p align="center">
  <img src="docs/images/logo.png" alt="H-DaC Logo" width="300">
</p>

# Head-Twitch Response Detection & Classification Tool

A comprehensive desktop application for detecting and analyzing Head-Twitch Responses (HTRs) in rodent behavioral videos using SLEAP pose-tracking data and machine learning. This tool provides a structured workflow from parameter tuning to model deployment.

---

## Prerequisites

**Software:**
- Python 3.9 (specifically – other versions may have compatibility issues)
- Windows 10/11, macOS 10.14+, or Linux

**Data Requirements:**
- **Pose tracking data:** SLEAP-generated H5 files (`.h5` format)
  - *Note: Support for other pose estimation formats (DeepLabCut, etc.) planned for future releases*
- **Camera view:** Top-down view of the animal required
- **Bodypart labels:** Videos must be tracked with these 5 specific keypoints:
  - Left Ear, Right Ear, Nose, Head, Back


<p align="center">
  <img src="docs/images/rat_labels_overhead_ex2.png" alt="Required Bodypart Labels" width="200">
  <br>
  <em>Required bodypart label placement (overhead view)</em>
</p>

---

## Quick Install

> **Important:** Use **Python 3.9** specifically. Other versions may cause dependency issues.

### Step 1: Get the Code

**Option A – With GitHub account (can receive updates):**
```bash
git clone https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification.git
cd Headtwitch_Detection_and_Classification
```

**Option B – Without GitHub account:**
1. Download ZIP from: [Download ZIP](https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification/archive/refs/heads/main.zip)
2. Extract and open a terminal in that folder

### Step 2: Set Up Environment

**Option A – Using Conda/Miniforge/Mambaforge (recommended):**
```bash
mamba create -n htr_env python=3.9 -y   # or: conda create -n htr_env python=3.9 -y
mamba activate htr_env                   # or: conda activate htr_env
pip install -r requirements.txt
```

**Option B – Using Python venv (requires Python 3.9 already installed):**
```bash
python -m venv htr_env
htr_env\Scripts\activate      # Windows
# source htr_env/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Step 3: Launch
```bash
python -m hdac
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not found | Use conda/mamba approach, or install Python 3.9 from [python.org](https://python.org) and check "Add to PATH" |
| Lockfile error in conda/mamba | Run terminal as Administrator, or run `mamba clean --locks` |
| App won't start on Python 3.10+ | Use Python 3.9 instead |

---

## Using H-DaC

### Scenario 1: [Ready to Predict HTRs](docs/workflow.md#deployment-fresh)
(Choose this scenario if you already have tuned detection parameters and a trained model) 
1. **(Jump to) New Project** → [Deployment Guide: Fresh Start](docs/workflow.md#deployment-fresh)
2. **(Jump to) Existing Project** → [Deployment Guide: Incremental Analysis](docs/workflow.md#deployment-incremental)

### Scenario 2: [Need to Prepare Your Detection System](docs/workflow.md#prepare-data)
(Choose this scenario if you're starting completely fresh or haven't finished tuning
parameters, labeling data, or training a model)
1. **(Jump to) Tune detection parameters** → [Parameter Tuning Guide](docs/parameter_tuning_guide.md)
2. **(Jump to) Label ground truth data** → [Ground Truth Labeling Guide](docs/workflow.md#prepare-data)
3. **(Jump to) Train your model** → [Model Training Guide](docs/workflow.md#train-model)

---

## How Does H-DaC Work?
**Understanding the detection methods:**

The tool uses **two complementary detection methods** to identify HTR events:

| *Ear Detector Approach* | *Head Detector Approach* |
|:--:|:--:|
| <img src="docs/screenshots/ear_detection_graphic.png" width="400"> | <img src="docs/screenshots/head_detection_graphic.png" width="400"> |
| *Real Ear Example* | *Real Head Example* |
| <img src="docs/screenshots/ear_detect.png" width="400"> | <img src="docs/screenshots/head_detect.png" width="400"> |

**Event Confidence Levels:**
- **Combined (Green)**: Detected by both methods → High confidence
- **Ear Only (Orange)**: Detected by ear method only → Medium confidence
- **Head Only (Red)**: Detected by head method only → Medium confidence

Events are classified using a trained XGBoost model that learns from user-labeled ground truth data.

👉 **[Read detailed methods documentation →](docs/detection_methods.md)**

### Need Help?
- **❓ Common issues and solutions** → [FAQ & Troubleshooting](docs/faq.md)
- **🐛 Report a bug** → [GitHub Issues](https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification/issues)
- **💬 Ask questions** → [GitHub Discussions](https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification/discussions)

---

## Documentation

Complete documentation available in the [`/docs`](docs/) directory:

| Guide | Status | Description |
|-------|--------|-------------|
| **[Setup Guide](docs/setup_guide.md)** | ✅ Available | Hardware setup, camera configuration, SLEAP tracking |
| **[Detection Methods](docs/detection_methods.md)** | ✅ Available | Detailed explanation of dual detection methods |
| **[Workflow Guide](docs/workflow.md)** | ✅ Available | Complete end-to-end workflow with decision points |
| **[Installation Guide](docs/installation.md)** | 🚧 Coming Soon | Detailed software installation options and troubleshooting |
| **[Parameter Tuning Guide](docs/parameter_tuning_guide.md)** | 🚧 Coming Soon | Understanding and adjusting detection parameters |
| **[FAQ & Troubleshooting](docs/faq.md)** | 🚧 Coming Soon | Common questions and issues |

📹 **Video Tutorials** - Coming soon!

---

## Configuration

### Node Mapping

SLEAP exports can have different node orderings. Configure via **Settings > Configure Node Mapping**.

Common formats:
- Format 1: Left Ear=0, Right Ear=1, Back=2, Nose=3, Head=4
- Format 2: Nose=0, Head=1, Left Ear=2, Right Ear=3, Back=4

### Parameter Files

Detection parameters can be saved and loaded as JSON files:
- **File > Export Parameters**: Save current parameter configuration
- **File > Import Parameters**: Load saved parameter configuration

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{htr_analysis_tool_2025,
  author = {Butcher, Grayson},
  title = {Head-Twitch Response Detection and Classification Tool},
  year = {2025},
  url = {https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs or request features via [GitHub Issues](https://github.com/GraysonButcher/Headtwitch_Detection_and_Classification/issues)
- Submit pull requests with improvements
- Share your parameter configurations or training data

---

**Status**: Active Development | **Version**: 3.0.0 | **Last Updated**: January 2025
