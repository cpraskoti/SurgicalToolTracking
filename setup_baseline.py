#!/usr/bin/env python3
"""
Surgical Tool Tracking Baseline Setup Script
============================================

This script sets up the reproducible baseline environment for surgical tool tracking
research on the CholecTrack20 dataset using Bot-SORT as the baseline tracker.

Based on the research document: "A Foundational Baseline for Advanced Surgical Tool Tracking on CholecTrack20"
Phase 1: Foundation and Reproduction

Author: PhD Research Project Setup
Date: 2025
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Run a shell command with error handling."""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"[OUT] {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_environment():
    """Check if the environment is properly set up."""
    project_root = Path(__file__).parent
    
    required_dirs = [
        project_root / "cholectrack20",
        project_root / "BoT-SORT",
        project_root / "surgical_tracking_env"
    ]
    
    print("[INFO] Checking environment setup...")
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.name} found")
        else:
            print(f"✗ {dir_path.name} missing")
            return False
    
    return True


def install_additional_requirements():
    """Install any additional requirements for the baseline."""
    project_root = Path(__file__).parent
    venv_python = project_root / "surgical_tracking_env" / "bin" / "python"
    
    additional_packages = [
        "pycocotools",  # For COCO format support
        "seaborn",      # For visualization
        "jupyter",      # For notebooks
        "ipykernel",    # For Jupyter kernels
    ]
    
    for package in additional_packages:
        run_command(
            f"{venv_python} -m pip install {package}",
            f"Installing {package}",
            check=False
        )


def create_directory_structure():
    """Create the project directory structure."""
    project_root = Path(__file__).parent
    
    dirs_to_create = [
        "data",
        "results", 
        "experiments",
        "models",
        "notebooks",
        "scripts",
        "configs"
    ]
    
    print("[INFO] Creating project directory structure...")
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")


def create_config_files():
    """Create configuration files for the baseline."""
    project_root = Path(__file__).parent
    
    # Create a basic configuration file for Bot-SORT on CholecTrack20
    config_content = """# Bot-SORT Configuration for CholecTrack20
# ===========================================

# Dataset Configuration
dataset:
  name: "CholecTrack20"
  root_path: "./data/cholectrack20"
  splits:
    train: "train"
    val: "val" 
    test: "test"
  
# Detector Configuration  
detector:
  model: "yolov7"
  weights: "yolov7.pt"
  conf_threshold: 0.5
  iou_threshold: 0.45
  
# Tracker Configuration
tracker:
  name: "bot_sort"
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
  frame_rate: 25
  
# Evaluation Configuration
evaluation:
  metrics: ["HOTA", "CLEAR", "Identity", "Count"]
  perspectives: ["visibility", "intracorporeal", "intraoperative"]
  
# Output Configuration
output:
  results_dir: "./results"
  save_videos: false
  save_images: false
"""
    
    config_path = project_root / "configs" / "baseline_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created baseline configuration: {config_path}")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Surgical Tool Tracking Baseline Setup")
    print("=" * 60)
    
    # Check if environment is set up
    if not check_environment():
        print("\n[ERROR] Environment not properly set up!")
        print("Please ensure you have run the setup commands first.")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Install additional requirements
    install_additional_requirements()
    
    # Create config files
    create_config_files()
    
    print("\n" + "=" * 60)
    print("BASELINE SETUP COMPLETE!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Download the CholecTrack20 dataset following the DUA requirements")
    print("2. Place the dataset in the ./data/cholectrack20/ directory")
    print("3. Run the baseline evaluation using Bot-SORT")
    print("4. Compare results with the benchmark in the research document")
    
    print("\nProject structure:")
    project_root = Path(__file__).parent
    for item in sorted(project_root.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            print(f"  {item.name}/")
    
    print(f"\nVirtual environment: {project_root}/surgical_tracking_env/")
    print("Activate with: source surgical_tracking_env/bin/activate")


if __name__ == "__main__":
    main()
