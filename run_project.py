
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    print("Checking dependencies...")
    try:
        import torch
        import cv2
        print("Core dependencies installed.")
    except ImportError as e:
        print(f"Missing core dependency: {e.name}")
        return False

    try:
        import roboflow
    except ImportError:
        print("Warning: roboflow not found. Skipping dataset download features.")
        
    return True

def main():
    if not check_dependencies():
        return

    ROOT = Path(__file__).resolve().parent
    WEIGHTS_PATH = ROOT / "runs/train/exp/weights/best.pt"  # Assumed output path after training
    
    # Check if weights exist. Note: The training script might save to a different 'exp' folder (exp, exp2, etc.)
    # So we might need to find the latest 'exp' folder or user can specify.
    # For now, we check if ANY match exists or prompt user.
    
    found_weights = None
    if os.path.exists(WEIGHTS_PATH):
        found_weights = WEIGHTS_PATH
    elif os.path.exists(ROOT / "best.pt"):
        found_weights = ROOT / "best.pt"
    else:
        # Check recursively in runs/train for best.pt
        import glob
        potential_weights = glob.glob(str(ROOT / "runs/train/**/best.pt"), recursive=True)
        if potential_weights:
            found_weights = potential_weights[-1] # Take the last one found, likely the latest
    
    if not found_weights:
        print("No trained weights found.")
        choice = input("Do you want to download the dataset and train the model now? (y/n): ")
        if choice.lower() == 'y':
            print("Launching training script...")
            try:
                subprocess.check_call([sys.executable, "train_model.py"])
                # Re-check for weights
                potential_weights = glob.glob(str(ROOT / "runs/train/**/best.pt"), recursive=True)
                if potential_weights:
                    found_weights = potential_weights[-1]
                else:
                     print("Training finished but no weights found. Something went wrong.")
                     return
            except subprocess.CalledProcessError:
                print("Training was interrupted or failed.")
                return
        else:
            print("Cannot run inference without weights.")
            return
            
    print(f"Using weights: {found_weights}")
    
    # Ask for source
    source = input(f"Enter path to image/video or '0' for webcam (default: {ROOT / 'data/images'}): ")
    if not source:
        source = str(ROOT / 'data/images')
    
    print(f"Running inference on {source}...")
    cmd = [
        sys.executable, "detect2.py",
        "--weights", str(found_weights),
        "--source", source,
        "--img", "640",
        "--data", str(ROOT / "data.yaml"),
        "--view-img" # Optional: view output
    ]
    subprocess.call(cmd)

if __name__ == "__main__":
    main()
