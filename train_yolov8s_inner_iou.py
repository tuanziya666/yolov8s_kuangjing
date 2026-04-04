import argparse
import os
import warnings
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

os.environ.setdefault("WANDB_MODE", "disabled")
SETTINGS.update(wandb=False, raytune=False)

warnings.filterwarnings("ignore")


DATA_YAML = Path("ultralytics/cfg/datasets/coco8.yaml")
MODEL_WEIGHTS = "yolov8s.pt"
DEFAULT_DEVICE = "0"
DEFAULT_INNER_RATIO = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8s model with Inner-IoU bbox regression loss.")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_WEIGHTS, help="Model weights or yaml.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="yolov8s_inner_iou", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--inner-ratio", type=float, default=DEFAULT_INNER_RATIO, help="Inner-IoU box scaling ratio.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided model path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["ULTRALYTICS_IOU_LOSS"] = "inner_iou"
    os.environ["ULTRALYTICS_INNER_IOU_RATIO"] = str(args.inner_ratio)

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "task": "detect",
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
    }
    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)
