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
MODEL_CFG = "ultralytics/cfg/models/v8/yolov8s_afpn.yaml"
PRETRAINED_WEIGHTS = "yolov8s.pt"
DEFAULT_DEVICE = "0"

CONFLICT_ENV_KEYS = [
    "ULTRALYTICS_CLS_LOSS",
    "ULTRALYTICS_QFL_BETA",
    "ULTRALYTICS_IOU_LOSS",
    "ULTRALYTICS_INNER_IOU_RATIO",
    "ULTRALYTICS_WCIOU_AC_LAMBDA",
    "ULTRALYTICS_WCIOU_AC_GAMMA",
    "ULTRALYTICS_SA_BOX_ENABLE",
    "ULTRALYTICS_TAL_REG_ENABLE",
    "ULTRALYTICS_AUX_HEAD_ENABLE",
    "ULTRALYTICS_USE_DIFFICULTY_SAMPLER",
    "ULTRALYTICS_SHAPE_LOSS_ENABLE",
    "ULTRALYTICS_HARD_BOX_ENABLE",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8s AFPN with ATFL classification loss and MPDIoU box regression loss."
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_CFG, help="Model yaml or checkpoint path.")
    parser.add_argument("--pretrained", default=PRETRAINED_WEIGHTS, help="Pretrained weights for yaml models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="yolov8s_afpn_atfl_mpdiou", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--cache", type=str2bool, default=False, help="Whether to cache the dataset.")
    parser.add_argument("--amp", type=str2bool, default=True, help="Whether to use AMP training.")
    parser.add_argument("--deterministic", type=str2bool, default=True, help="Whether to use deterministic training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided checkpoint path.")
    return parser.parse_args()


def configure_env() -> None:
    for key in CONFLICT_ENV_KEYS:
        os.environ.pop(key, None)

    os.environ["ULTRALYTICS_CLS_LOSS"] = "atfl"
    os.environ["ULTRALYTICS_IOU_LOSS"] = "mpdiou"


def main():
    args = parse_args()
    configure_env()

    print("model =", args.model)
    print("pretrained =", args.pretrained)
    print("data =", args.data)
    print("epochs =", args.epochs)
    print("batch =", args.batch)
    print("imgsz =", args.imgsz)
    print("device =", args.device)
    print("optimizer =", args.optimizer)
    print("seed =", args.seed)
    print("patience =", args.patience)
    print("cls_loss =", os.getenv("ULTRALYTICS_CLS_LOSS"))
    print("iou_loss =", os.getenv("ULTRALYTICS_IOU_LOSS"))

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
        "optimizer": args.optimizer,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "patience": args.patience,
        "cache": args.cache,
        "amp": args.amp,
        "save": True,
        "val": True,
        "plots": True,
        "verbose": True,
    }

    if str(args.model).lower().endswith((".yaml", ".yml")) and args.pretrained:
        train_kwargs["pretrained"] = args.pretrained
    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
