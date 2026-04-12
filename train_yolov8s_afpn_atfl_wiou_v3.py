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
    "ULTRALYTICS_SHAPE_IOU_SCALE",
    "ULTRALYTICS_WIOU_V3_MOMENTUM",
    "ULTRALYTICS_WIOU_V3_ALPHA",
    "ULTRALYTICS_WIOU_V3_DELTA",
    "ULTRALYTICS_WCIOU_AC_LAMBDA",
    "ULTRALYTICS_WCIOU_AC_GAMMA",
    "ULTRALYTICS_SA_BOX_ENABLE",
    "ULTRALYTICS_SA_SMALL_ALPHA",
    "ULTRALYTICS_SA_ELONG_BETA",
    "ULTRALYTICS_SA_CLASS_GAMMA",
    "ULTRALYTICS_SA_SMALL_AREA_THR",
    "ULTRALYTICS_SA_ELONG_RATIO_THR",
    "ULTRALYTICS_SA_TARGET_CLASS_IDS",
    "ULTRALYTICS_TAL_REG_ENABLE",
    "ULTRALYTICS_TAL_REG_START_EPOCH",
    "ULTRALYTICS_TAL_REG_EMA_DECAY",
    "ULTRALYTICS_TAL_REG_GAIN",
    "ULTRALYTICS_TAL_REG_THRESHOLD",
    "ULTRALYTICS_HARD_BOX_ENABLE",
    "ULTRALYTICS_HARD_BOX_CLASS_ALPHA",
    "ULTRALYTICS_HARD_BOX_SMALL_H1",
    "ULTRALYTICS_HARD_BOX_SMALL_H2",
    "ULTRALYTICS_HARD_BOX_HEIGHT_W1",
    "ULTRALYTICS_HARD_BOX_HEIGHT_W2",
    "ULTRALYTICS_HARD_BOX_USE_RATIO",
    "ULTRALYTICS_HARD_BOX_RATIO_T1",
    "ULTRALYTICS_HARD_BOX_RATIO_T2",
    "ULTRALYTICS_HARD_BOX_RATIO_W1",
    "ULTRALYTICS_HARD_BOX_RATIO_W2",
    "ULTRALYTICS_HARD_BOX_MAX_WEIGHT",
    "ULTRALYTICS_HARD_BOX_TARGET_CLASS_IDS",
    "ULTRALYTICS_SHAPE_LOSS_ENABLE",
    "ULTRALYTICS_SHAPE_LOSS_LAMBDA",
    "ULTRALYTICS_SHAPE_BETA",
    "ULTRALYTICS_SHAPE_CLASS_GAMMA",
    "ULTRALYTICS_SHAPE_THIN_GAMMA",
    "ULTRALYTICS_SHAPE_ELONG_GAMMA",
    "ULTRALYTICS_SHAPE_THIN_SIDE_THR",
    "ULTRALYTICS_SHAPE_ELONG_RATIO_THR",
    "ULTRALYTICS_SHAPE_TARGET_CLASS_IDS",
    "ULTRALYTICS_AUX_HEAD_ENABLE",
    "ULTRALYTICS_AUX_HEAD_GAIN",
    "ULTRALYTICS_AUX_SMALL_AREA_THR",
    "ULTRALYTICS_AUX_TARGET_CLASS_IDS",
    "ULTRALYTICS_USE_DIFFICULTY_SAMPLER",
    "ULTRALYTICS_QUALITY_HEAD_ENABLE",
    "ULTRALYTICS_QUALITY_HEAD_LEVELS",
    "ULTRALYTICS_QUALITY_HEAD_LAMBDA",
    "ULTRALYTICS_QUALITY_HEAD_SCORE_MODE",
    "ULTRALYTICS_QUALITY_HEAD_ALPHA",
    "ULTRALYTICS_QUALITY_HEAD_DRILL_WEIGHT_ENABLE",
    "ULTRALYTICS_QUALITY_HEAD_DRILL_WEIGHT_REFINE",
    "ULTRALYTICS_QUALITY_HEAD_TARGET_CLASS_IDS",
    "ULTRALYTICS_QUALITY_HEAD_DRILL_BASE_WEIGHT",
    "ULTRALYTICS_QUALITY_HEAD_SMALL_H1",
    "ULTRALYTICS_QUALITY_HEAD_SMALL_H2",
    "ULTRALYTICS_QUALITY_HEAD_DRILL_SMALL_W1",
    "ULTRALYTICS_QUALITY_HEAD_DRILL_SMALL_W2",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8s AFPN with ATFL classification loss and Wise-IoU v3 box regression loss."
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_CFG, help="Model yaml or checkpoint path.")
    parser.add_argument("--pretrained", default=PRETRAINED_WEIGHTS, help="Pretrained weights for yaml models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="yolov8s_afpn_atfl_wiou_v3", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--wiou-momentum", type=float, default=1e-2, help="Running mean momentum for Wise-IoU v3.")
    parser.add_argument("--wiou-alpha", type=float, default=1.7, help="Alpha for Wise-IoU v3 non-monotonic focusing.")
    parser.add_argument("--wiou-delta", type=float, default=2.7, help="Delta for Wise-IoU v3 non-monotonic focusing.")
    parser.add_argument("--cache", type=str2bool, default=False, help="Whether to cache the dataset.")
    parser.add_argument("--amp", type=str2bool, default=True, help="Whether to use AMP training.")
    parser.add_argument("--deterministic", type=str2bool, default=True, help="Whether to use deterministic training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided checkpoint path.")
    return parser.parse_args()


def configure_env(args) -> None:
    for key in CONFLICT_ENV_KEYS:
        os.environ.pop(key, None)

    os.environ["ULTRALYTICS_CLS_LOSS"] = "atfl"
    os.environ["ULTRALYTICS_IOU_LOSS"] = "wiou_v3"
    os.environ["ULTRALYTICS_WIOU_V3_MOMENTUM"] = str(args.wiou_momentum)
    os.environ["ULTRALYTICS_WIOU_V3_ALPHA"] = str(args.wiou_alpha)
    os.environ["ULTRALYTICS_WIOU_V3_DELTA"] = str(args.wiou_delta)


def main():
    args = parse_args()
    configure_env(args)

    print("model =", args.model)
    print("pretrained =", args.pretrained)
    print("data =", args.data)
    print("epochs =", args.epochs)
    print("batch =", args.batch)
    print("imgsz =", args.imgsz)
    print("device =", args.device)
    print("optimizer =", args.optimizer)
    print("seed =", args.seed)
    print("cls_loss =", os.getenv("ULTRALYTICS_CLS_LOSS"))
    print("iou_loss =", os.getenv("ULTRALYTICS_IOU_LOSS"))
    print("wiou_momentum =", os.getenv("ULTRALYTICS_WIOU_V3_MOMENTUM"))
    print("wiou_alpha =", os.getenv("ULTRALYTICS_WIOU_V3_ALPHA"))
    print("wiou_delta =", os.getenv("ULTRALYTICS_WIOU_V3_DELTA"))

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
