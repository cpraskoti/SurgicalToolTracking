# Surgical Tool Tracking - Reproducible Baseline Setup
## Project Structure

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
- **PyTorch 2.8.0**
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


## 📋 Next Steps

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
source envname/bin/activate

# Run Bot-SORT baseline evaluation
cd BoT-SORT
python tools/track.py --dataset_config ../configs/baseline_config.yaml

# Evaluate using TrackEval
cd ../cholectrack20/TrackEval
python scripts/run_mot_challenge.py --BENCHMARK CholecTrack20
```

## Roadmap

### Phase 1: Foundation and Reproduction 
- [x] Set up Bot-SORT baseline (best available alternative)
- [x] Configure TrackEval library for CholecTrack20
- [ ] Download CholecTrack20 dataset
- [ ] Reproduce baseline HOTA scores


## 📚 References

1. **CholecTrack20 Paper**: [arXiv:2312.07352](https://arxiv.org/abs/2312.07352)
2. **Bot-SORT Paper**: [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)
3. **SurgiTrack Paper**: [arXiv:2405.20333](https://arxiv.org/abs/2405.20333)
