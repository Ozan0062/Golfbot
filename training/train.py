#train YOLO models for GolfBot

import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

TRAINING_CONFIGS = {
    "field": {
        "dataset": "datasets/field/data.yaml", 
        "epochs": 100,
        "name": "field_model",
    },
    "objects": {
        "dataset": "datasets/objects/data.yaml", 
        "epochs": 150,
        "name": "objects_model",
    },
}

#settings
BASE_MODEL = "yolo11n.pt"
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT_DIR = "training_output"




def train(config_name):
    if config_name not in TRAINING_CONFIGS:
        print(f"Unknown config '{config_name}'. Choose: {list(TRAINING_CONFIGS.keys())}")
        sys.exit(1)

    cfg = TRAINING_CONFIGS[config_name]
    dataset_path = Path(cfg["dataset"])

    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}")
        print(f"Download from Roboflow and unzip to: {dataset_path.parent}/")
        sys.exit(1)

    print(f"\n{'═' * 50}")
    print(f"  Training: {config_name}")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Epochs:   {cfg['epochs']}")
    print(f"  Model:    {BASE_MODEL}")
    print(f"{'═' * 50}\n")

    #load
    model = YOLO(BASE_MODEL, task="detect")

    #train
    model.train(
        data=str(dataset_path),
        epochs=cfg["epochs"],
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=cfg["name"],
        exist_ok=True,     
        patience=50,       
        plots=True,       
    )

    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    
    print("\nExporting to ONNX...")
    model.export(format="onnx")

  
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.onnx"
    if config_name == "field":
        dst = Path("models/best_field.onnx")
    else:
        dst = Path("models/best_objects.onnx")

    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(best_pt, dst)
    print(f"\nDone! Best model copied to: {dst}")
    print(f"Training results saved to: {PROJECT_DIR}/{cfg['name']}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python training/train.py <field|objects>")
        sys.exit(1)

    train(sys.argv[1])