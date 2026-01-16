
import os
import sys
import argparse
from pathlib import Path
from roboflow import Roboflow

# Add current directory to sys.path to ensure modules can be imported
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def download_dataset(api_key, workspace, project_name, version_number, location):
    print(f"Downloading dataset to {location}...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version_number).download("yolov5", location=location)
    return dataset

def train(data_yaml, weights, epochs=3, batch_size=16, img_size=640):
    print(f"Starting training with data={data_yaml}, weights={weights}, epochs={epochs}...")
    # Using subprocess to call train.py to avoid complex import issues if train.py isn't perfectly modular
    import subprocess
    cmd = [
        sys.executable, "train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--cache"
    ]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default="4TFLqpycRN0FG5gHvY2z", help="Roboflow API Key")
    parser.add_argument("--workspace", type=str, default="object-detection-vpvcm", help="Roboflow Workspace")
    parser.add_argument("--project", type=str, default="nutracal-food-detection", help="Roboflow Project")
    parser.add_argument("--version", type=int, default=5, help="Dataset Version")
    parser.add_argument("--download-only", action="store_true", help="Download dataset and exit")
    
    args = parser.parse_args()

    # Define dataset location relative to ROOT
    dataset_location = str(ROOT / "NutraCal-Food-Detection-5")
    
    if not os.path.exists(dataset_location):
        try:
            download_dataset(args.api_key, args.workspace, args.project, args.version, dataset_location)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please check your API key or internet connection.")
            sys.exit(1)
    else:
        print(f"Dataset found at {dataset_location}")

    if not args.download_only:
        data_yaml = os.path.join(dataset_location, "data.yaml")
        # Ensure data.yaml exists
        if not os.path.exists(data_yaml):
             print(f"Error: {data_yaml} not found. Training cannot proceed.")
             sys.exit(1)
             
        # Use yolov5s.pt as starting weights if available, else download or use a placeholder
        weights = "yolov5s.pt" 
        
        train(data_yaml, weights)
