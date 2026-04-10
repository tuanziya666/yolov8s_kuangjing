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
DEFAULT_INNER_RATIO = 0.8

CONFLICT_ENV_KEYS = [
    "ULTRALYTICS_CLS_LOSS",
    "ULTRALYTICS_QFL_BETA",
    "ULTRALYTICS_IOU_LOSS",
    "ULTRALYTICS_INNER_IOU_RATIO",
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
    "ULTRALYTICS_DS_DRILL_PIPE_BONUS",
    "ULTRALYTICS_DS_COAL_MINER_BONUS",
    "ULTRALYTICS_DS_LOW_LIGHT_SCALE",
    "ULTRALYTICS_DS_ELONGATED_SCALE",
    "ULTRALYTICS_DS_MAX_SAMPLE_WEIGHT",
    "ULTRALYTICS_DS_ELONGATED_RATIO_THR",
    "ULTRALYTICS_DS_LOW_LIGHT_MEAN_THR",
    "ULTRALYTICS_DS_DRILL_PIPE_CLASS_IDS",
    "ULTRALYTICS_DS_COAL_MINER_CLASS_IDS",
    "ULTRALYTICS_DS_ATTR_CSV",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8s AFPN with ATFL classification, Inner-IoU regression, and sample-aware box/DFL weighting."
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_CFG, help="Model yaml or checkpoint path.")
    parser.add_argument("--pretrained", default=PRETRAINED_WEIGHTS, help="Pretrained weights for yaml models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="yolov8s_afpn_atfl_inneriou_sa_box", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--inner-ratio", type=float, default=DEFAULT_INNER_RATIO, help="Inner-IoU box scaling ratio.")
    parser.add_argument("--sa-small-alpha", type=float, default=0.5, help="Extra weight for small-area positives.")
    parser.add_argument("--sa-elong-beta", type=float, default=0.7, help="Extra weight for elongated positives.")
    parser.add_argument("--sa-class-gamma", type=float, default=0.5, help="Extra weight for target classes.")
    parser.add_argument(
        "--sa-small-area-thr",
        type=float,
        default=0.012521,
        help="Normalized area threshold used to define small positives.",
    )
    parser.add_argument(
        "--sa-elong-ratio-thr",
        type=float,
        default=3.0,
        help="Aspect-ratio threshold used to define elongated positives.",
    )
    parser.add_argument(
        "--sa-target-class-ids",
        default="2",
        help="Comma-separated target class ids for sample-aware weighting.",
    )
    parser.add_argument("--cache", type=str2bool, default=False, help="Whether to cache the dataset.")
    parser.add_argument("--amp", type=str2bool, default=True, help="Whether to use AMP training.")
    parser.add_argument("--deterministic", type=str2bool, default=True, help="Whether to use deterministic training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided checkpoint path.")
    return parser.parse_args()


def configure_env(args) -> None:
    for key in CONFLICT_ENV_KEYS:
        os.environ.pop(key, None)

    os.environ["ULTRALYTICS_CLS_LOSS"] = "atfl"
    os.environ["ULTRALYTICS_IOU_LOSS"] = "inner_iou"
    os.environ["ULTRALYTICS_INNER_IOU_RATIO"] = str(args.inner_ratio)

    os.environ["ULTRALYTICS_SA_BOX_ENABLE"] = "1"
    os.environ["ULTRALYTICS_SA_SMALL_ALPHA"] = str(args.sa_small_alpha)
    os.environ["ULTRALYTICS_SA_ELONG_BETA"] = str(args.sa_elong_beta)
    os.environ["ULTRALYTICS_SA_CLASS_GAMMA"] = str(args.sa_class_gamma)
    os.environ["ULTRALYTICS_SA_SMALL_AREA_THR"] = str(args.sa_small_area_thr)
    os.environ["ULTRALYTICS_SA_ELONG_RATIO_THR"] = str(args.sa_elong_ratio_thr)
    os.environ["ULTRALYTICS_SA_TARGET_CLASS_IDS"] = str(args.sa_target_class_ids)

    os.environ["ULTRALYTICS_HARD_BOX_ENABLE"] = "0"
    os.environ["ULTRALYTICS_SHAPE_LOSS_ENABLE"] = "0"
    os.environ["ULTRALYTICS_TAL_REG_ENABLE"] = "0"
    os.environ["ULTRALYTICS_AUX_HEAD_ENABLE"] = "0"
    os.environ["ULTRALYTICS_USE_DIFFICULTY_SAMPLER"] = "0"


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
    print("inner_ratio =", os.getenv("ULTRALYTICS_INNER_IOU_RATIO"))
    print("sa_box_enable =", os.getenv("ULTRALYTICS_SA_BOX_ENABLE"))
    print("sa_small_alpha =", os.getenv("ULTRALYTICS_SA_SMALL_ALPHA"))
    print("sa_elong_beta =", os.getenv("ULTRALYTICS_SA_ELONG_BETA"))
    print("sa_class_gamma =", os.getenv("ULTRALYTICS_SA_CLASS_GAMMA"))
    print("sa_small_area_thr =", os.getenv("ULTRALYTICS_SA_SMALL_AREA_THR"))
    print("sa_elong_ratio_thr =", os.getenv("ULTRALYTICS_SA_ELONG_RATIO_THR"))
    print("sa_target_class_ids =", os.getenv("ULTRALYTICS_SA_TARGET_CLASS_IDS"))
    print("note = sample-aware weighting affects both IoU loss and DFL loss")

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
