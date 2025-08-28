# Surgical Tool Tracking - Reproducible Baseline Setup

This repository contains the reproducible baseline setup for a PhD research project on surgical tool tracking using the CholecTrack20 dataset, following the Phase 1 recommendations from the research document **"A Foundational Baseline for Advanced Surgical Tool Tracking on CholecTrack20"**.

## 🎯 Research Objectives

- **Primary Goal**: Improve HOTA scores from <45% to >80% for surgical tool tracking
- **Phase 1**: Establish reproducible baseline using best-performing tracker (Bot-SORT)
- **Dataset**: CholecTrack20 multi-perspective surgical tool tracking dataset
- **Evaluation**: HOTA, CLEAR MOT, Identity, and Count metrics across 3 tracking perspectives

## 📋 Current Setup Status

✅ **Completed:**
- Virtual environment setup with Python 3.10
- Bot-SORT baseline tracker cloned and configured
- TrackEval library installed and configured for CholecTrack20
- Project directory structure created
- Dependencies installed (PyTorch, OpenCV, etc.)

⏳ **Pending:**
- CholecTrack20 dataset download (requires DUA acceptance)
- Baseline evaluation runs
- Performance benchmarking

## 🛠 Environment Setup

### Prerequisites
- Ubuntu/Linux environment 
- Python 3.10+
- Git
- Internet connection for package downloads

### Quick Start

1. **Activate the environment:**
```bash
cd /home/tachyon/fluidic/SurgicalTracking
source surgical_tracking_env/bin/activate
```

2. **Run the setup script:**
```bash
python setup_baseline.py
```

3. **Verify installation:**
```bash
python -c "import cv2, torch, trackeval; print('Environment ready!')"
```

## 📁 Project Structure

```
SurgicalTracking/
├── README.md                          # This file
├── setup_baseline.py                  # Automated setup script
├── surgical_tracking_env/             # Python virtual environment
├── BoT-SORT/                          # Bot-SORT tracker (baseline)
├── cholectrack20/                     # CholecTrack20 dataset repository
│   ├── TrackEval/                     # Adapted evaluation library
│   ├── utils/                         # Dataset utilities
│   └── README.md                      # Dataset documentation
├── data/                              # Dataset storage (to be populated)
├── results/                           # Evaluation results
├── experiments/                       # Experiment configurations
├── models/                            # Trained model weights
├── notebooks/                         # Jupyter notebooks
├── scripts/                           # Utility scripts
└── configs/                           # Configuration files
```

## 📊 Expected Baseline Performance

Based on the CholecTrack20 benchmark results, Bot-SORT achieves:

| Tracking Perspective | HOTA ↑ | DetA ↑ | AssA ↑ | MOTA ↑ | IDF1 ↑ |
|---------------------|--------|--------|--------|--------|--------|
| **Visibility**      | 44.7   | 70.8   | 28.7   | 72.0   | 41.4   |
| **Intracorporeal**   | 27.0   | 70.7   | 10.4   | 70.0   | 18.9   |
| **Intraoperative**   | 17.4   | 70.7   | 4.4    | 69.6   | 10.2   |

## 🔧 Installed Dependencies

### Core Packages
- **PyTorch 2.8.0** (CPU version)
- **OpenCV 4.12.0** - Computer vision operations
- **TrackEval 1.0.dev1** - Evaluation metrics (CholecTrack20 adapted)
- **Motmetrics 1.4.0** - MOT evaluation utilities

### Bot-SORT Dependencies
- **NumPy, SciPy** - Numerical computations
- **Matplotlib, Seaborn** - Visualization
- **Scikit-learn, Scikit-image** - Machine learning and image processing
- **FilterPy** - Kalman filtering
- **LAP** - Linear assignment problems
- **YACS, EasyDict** - Configuration management

### Research Tools
- **Jupyter** - Interactive notebooks
- **TensorBoard** - Experiment logging
- **Loguru** - Advanced logging
- **TQDM** - Progress bars

## 📋 Next Steps (Phase 1 Continuation)

### 1. Dataset Acquisition
```bash
# 1. Read the Data Use Agreement
cat cholectrack20/DUA.md

# 2. Complete the dataset request form (available from March 25, 2025)
# https://docs.google.com/forms/d/e/1FAIpQLSdewhAi0vGmZj5DLOMWdLf85BhUtTedS28YzvHS58ViwuEX5w/viewform

# 3. Download from Synapse.org with your access key
# https://www.synapse.org/Synapse:syn53182642/wiki/
```

### 2. Dataset Preparation
```bash
# Extract and organize the dataset
mkdir -p data/cholectrack20
# Follow the dataset structure guidelines from cholectrack20/README.md
```

### 3. Baseline Evaluation
```bash
# Activate environment
source surgical_tracking_env/bin/activate

# Run Bot-SORT baseline evaluation
cd BoT-SORT
python tools/track.py --dataset_config ../configs/baseline_config.yaml

# Evaluate using TrackEval
cd ../cholectrack20/TrackEval
python scripts/run_mot_challenge.py --BENCHMARK CholecTrack20
```

### 4. Performance Analysis
```bash
# Generate performance reports
python -c "
import trackeval
# Load results and generate HOTA, CLEAR, Identity metrics
# Compare with benchmark results from research document
"
```

## 🎯 Research Roadmap

### Phase 1: Foundation and Reproduction ✅
- [x] Clone SurgiTrack and CholecTrack20 repositories  
- [x] Set up Bot-SORT baseline (best available alternative)
- [x] Configure TrackEval library for CholecTrack20
- [ ] Download CholecTrack20 dataset
- [ ] Reproduce baseline HOTA scores

### Phase 2: Robust Feature Learning (Future)
- [ ] Implement advanced data augmentation for adverse conditions
- [ ] Develop domain-robust detection features  
- [ ] Enhance DetA scores under visual challenges

### Phase 3: Context-Aware Re-identification (Future)
- [ ] Implement operator-aware re-identification
- [ ] Integrate surgical phase information
- [ ] Improve AssA scores and reduce identity switches

## 📝 Key Considerations

### SurgiTrack Status
- **Note**: The original SurgiTrack code is marked as "Coming soon!" in their repository
- **Alternative**: Using Bot-SORT as baseline (best performing available tracker: 44.7% HOTA)
- **Future**: Will migrate to SurgiTrack when code becomes available

### Dataset Licensing
- CholecTrack20 is released under CC-BY-NC-SA 4.0 license
- Research use only (non-commercial)
- Requires completion of Data Use Agreement (DUA)
- Dataset requests accepted from March 25, 2025

### Computing Requirements
- **CPU**: Multi-core recommended for tracking operations
- **Memory**: 8GB+ RAM recommended for dataset processing
- **Storage**: ~50GB for full dataset and results
- **GPU**: Optional but recommended for training (future phases)

## 🔍 Troubleshooting

### Environment Issues
```bash
# Reinstall environment if needed
rm -rf surgical_tracking_env
python3 -m venv surgical_tracking_env
source surgical_tracking_env/bin/activate
pip install --upgrade pip
# Re-run dependency installation
```

### Import Errors
```bash
# Verify package installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import trackeval; print('TrackEval: OK')"
```

### Dataset Path Issues
```bash
# Check dataset structure
ls -la data/cholectrack20/
# Should contain: train/, val/, test/ directories
```

## 📚 References

1. **Research Document**: "A Foundational Baseline for Advanced Surgical Tool Tracking on CholecTrack20"
2. **CholecTrack20 Paper**: [arXiv:2312.07352](https://arxiv.org/abs/2312.07352)
3. **Bot-SORT Paper**: [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)
4. **SurgiTrack Paper**: [arXiv:2405.20333](https://arxiv.org/abs/2405.20333)

## 📄 License

This baseline setup follows the licensing terms of:
- CholecTrack20: CC-BY-NC-SA 4.0 (research use)
- Bot-SORT: MIT License
- TrackEval: MIT License

## 👥 Acknowledgments

- CAMMA Research Group (University of Strasbourg) for CholecTrack20 dataset
- Bot-SORT authors for the baseline tracking framework
- TrackEval authors for evaluation metrics library

---

**Status**: Phase 1 Setup Complete ✅ | **Next**: Dataset Download & Baseline Evaluation
