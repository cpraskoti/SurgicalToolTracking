#!/usr/bin/env python3
"""
Setup Verification Script
=========================
Verifies that the surgical tracking baseline setup is correctly configured.
"""

import sys
import importlib


def check_package(package_name, display_name=None):
    """Check if a package can be imported and get its version."""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {display_name}: NOT FOUND")
        return False


def main():
    """Main verification function."""
    print("🔍 Verifying Surgical Tracking Environment Setup")
    print("=" * 50)
    
    packages = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("trackeval", "TrackEval"),
        ("motmetrics", "MOTMetrics"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package, display in packages:
        if check_package(package, display):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Setup Status: {success_count}/{total_count} packages available")
    
    if success_count == total_count:
        print("🎉 Environment setup is COMPLETE!")
        print("\n📋 Next Steps:")
        print("1. Download CholecTrack20 dataset (requires DUA acceptance)")
        print("2. Run baseline evaluation: python scripts/run_baseline_evaluation.py")
        return True
    else:
        print("❌ Environment setup is INCOMPLETE!")
        print("Please check missing packages and reinstall if needed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
