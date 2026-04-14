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
    "ULTRALYTICS_SHAPE_IOU_SCALE",
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
    "ULTRALYTICS_DLQ_HEAD_ENABLE",
    "ULTRALYTICS_DLQ_HEAD_LEVELS",
    "ULTRALYTICS_DLQ_HEAD_LAMBDA",
    "ULTRALYTICS_DLQ_HEAD_SCORE_MODE",
    "ULTRALYTICS_DLQ_HEAD_ALPHA",
    "ULTRALYTICS_DLQ_HEAD_DRILL_WEIGHT_ENABLE",
    "ULTRALYTICS_DLQ_HEAD_DRILL_WEIGHT_REFINE",
    "ULTRALYTICS_DLQ_HEAD_TARGET_CLASS_IDS",
    "ULTRALYTICS_DLQ_HEAD_DRILL_BASE_WEIGHT",
    "ULTRALYTICS_DLQ_HEAD_SMALL_H1",
    "ULTRALYTICS_DLQ_HEAD_SMALL_H2",
    "ULTRALYTICS_DLQ_HEAD_DRILL_SMALL_W1",
    "ULTRALYTICS_DLQ_HEAD_DRILL_SMALL_W2",
    "ULTRALYTICS_DLR_HEAD_ENABLE",
    "ULTRALYTICS_DLR_HEAD_LEVELS",
    "ULTRALYTICS_DLR_HEAD_LAMBDA",
    "ULTRALYTICS_DLR_HEAD_SCORE_MODE",
    "ULTRALYTICS_DLR_HEAD_ALPHA",
    "ULTRALYTICS_DLR_HEAD_DRILL_WEIGHT_ENABLE",
    "ULTRALYTICS_DLR_HEAD_DRILL_WEIGHT_REFINE",
    "ULTRALYTICS_DLR_HEAD_TARGET_CLASS_IDS",
    "ULTRALYTICS_DLR_HEAD_DRILL_BASE_WEIGHT",
    "ULTRALYTICS_DLR_HEAD_SMALL_H1",
    "ULTRALYTICS_DLR_HEAD_SMALL_H2",
    "ULTRALYTICS_DLR_HEAD_DRILL_SMALL_W1",
    "ULTRALYTICS_DLR_HEAD_DRILL_SMALL_W2",
    "ULTRALYTICS_DGLR_HEAD_ENABLE",
    "ULTRALYTICS_DGLR_HEAD_LEVELS",
    "ULTRALYTICS_DGLR_HEAD_LAMBDA",
    "ULTRALYTICS_DGLR_HEAD_SCORE_MODE",
    "ULTRALYTICS_DGLR_HEAD_ALPHA",
    "ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_ENABLE",
    "ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_REFINE",
    "ULTRALYTICS_DGLR_HEAD_TARGET_CLASS_IDS",
    "ULTRALYTICS_DGLR_HEAD_DRILL_BASE_WEIGHT",
    "ULTRALYTICS_DGLR_HEAD_SMALL_H1",
    "ULTRALYTICS_DGLR_HEAD_SMALL_H2",
    "ULTRALYTICS_DGLR_HEAD_DRILL_SMALL_W1",
    "ULTRALYTICS_DGLR_HEAD_DRILL_SMALL_W2",
]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8s AFPN with ATFL + Inner-IoU + Drill-pipe Glare-aware Localization Refinement Head (DGLR-Head)."
        )
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_CFG, help="Model yaml or checkpoint path.")
    parser.add_argument("--pretrained", default=PRETRAINED_WEIGHTS, help="Pretrained weights for yaml models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="yolov8s_afpn_atfl_inneriou_dglr_head", help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--save-period", type=int, default=-1, help="Save epoch checkpoints every N epochs.")
    parser.add_argument("--cos-lr", type=str2bool, default=False, help="Use cosine learning-rate schedule.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in the final N epochs.")
    parser.add_argument("--inner-ratio", type=float, default=DEFAULT_INNER_RATIO, help="Inner-IoU box scaling ratio.")
    parser.add_argument("--dglr-levels", default="p3p4", help="DGLR head levels: 'p3p4' (recommended) or 'all'.")
    parser.add_argument("--dglr-lambda", type=float, default=0.20, help="Weight of the DGLR quality loss term.")
    parser.add_argument(
        "--dglr-score-mode",
        default="mul",
        choices=["mul", "pow"],
        help="How cls and quality scores are fused during inference.",
    )
    parser.add_argument(
        "--dglr-alpha",
        type=float,
        default=0.6,
        help="Alpha for score fusion when dglr-score-mode=pow.",
    )
    parser.add_argument(
        "--use-drill-quality-weight",
        type=str2bool,
        default=True,
        help="Apply drill_pipe-aware weighting to DGLR quality loss only.",
    )
    parser.add_argument(
        "--drill-quality-refine",
        type=str2bool,
        default=True,
        help="Use 1.3 for drill_pipe with h < 0.06 and 1.15 for 0.06 <= h < 0.09.",
    )
    parser.add_argument(
        "--dglr-target-class-ids",
        default="2",
        help="Comma-separated target class ids used by drill-aware DGLR weighting.",
    )
    parser.add_argument("--drill-quality-base-weight", type=float, default=1.2, help="Flat drill_pipe DGLR weight.")
    parser.add_argument("--drill-quality-small-h1", type=float, default=0.06, help="Small-target height threshold.")
    parser.add_argument("--drill-quality-small-h2", type=float, default=0.09, help="Secondary height threshold.")
    parser.add_argument("--drill-quality-small-w1", type=float, default=1.3, help="Weight for very small drill_pipe.")
    parser.add_argument("--drill-quality-small-w2", type=float, default=1.15, help="Weight for medium-small drill_pipe.")
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

    os.environ["ULTRALYTICS_DGLR_HEAD_ENABLE"] = "1"
    os.environ["ULTRALYTICS_DGLR_HEAD_LEVELS"] = str(args.dglr_levels)
    os.environ["ULTRALYTICS_DGLR_HEAD_LAMBDA"] = str(args.dglr_lambda)
    os.environ["ULTRALYTICS_DGLR_HEAD_SCORE_MODE"] = str(args.dglr_score_mode)
    os.environ["ULTRALYTICS_DGLR_HEAD_ALPHA"] = str(args.dglr_alpha)
    os.environ["ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_ENABLE"] = "1" if args.use_drill_quality_weight else "0"
    os.environ["ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_REFINE"] = "1" if args.drill_quality_refine else "0"
    os.environ["ULTRALYTICS_DGLR_HEAD_TARGET_CLASS_IDS"] = str(args.dglr_target_class_ids)
    os.environ["ULTRALYTICS_DGLR_HEAD_DRILL_BASE_WEIGHT"] = str(args.drill_quality_base_weight)
    os.environ["ULTRALYTICS_DGLR_HEAD_SMALL_H1"] = str(args.drill_quality_small_h1)
    os.environ["ULTRALYTICS_DGLR_HEAD_SMALL_H2"] = str(args.drill_quality_small_h2)
    os.environ["ULTRALYTICS_DGLR_HEAD_DRILL_SMALL_W1"] = str(args.drill_quality_small_w1)
    os.environ["ULTRALYTICS_DGLR_HEAD_DRILL_SMALL_W2"] = str(args.drill_quality_small_w2)

    os.environ["ULTRALYTICS_SA_BOX_ENABLE"] = "0"
    os.environ["ULTRALYTICS_HARD_BOX_ENABLE"] = "0"
    os.environ["ULTRALYTICS_TAL_REG_ENABLE"] = "0"
    os.environ["ULTRALYTICS_AUX_HEAD_ENABLE"] = "0"
    os.environ["ULTRALYTICS_USE_DIFFICULTY_SAMPLER"] = "0"
    os.environ["ULTRALYTICS_SHAPE_LOSS_ENABLE"] = "0"


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
    print("save_period =", args.save_period)
    print("cos_lr =", args.cos_lr)
    print("close_mosaic =", args.close_mosaic)
    print("cls_loss =", os.getenv("ULTRALYTICS_CLS_LOSS"))
    print("iou_loss =", os.getenv("ULTRALYTICS_IOU_LOSS"))
    print("inner_ratio =", os.getenv("ULTRALYTICS_INNER_IOU_RATIO"))
    print("dglr_head_enable =", os.getenv("ULTRALYTICS_DGLR_HEAD_ENABLE"))
    print("dglr_head_levels =", os.getenv("ULTRALYTICS_DGLR_HEAD_LEVELS"))
    print("dglr_lambda =", os.getenv("ULTRALYTICS_DGLR_HEAD_LAMBDA"))
    print("dglr_score_mode =", os.getenv("ULTRALYTICS_DGLR_HEAD_SCORE_MODE"))
    print("dglr_alpha =", os.getenv("ULTRALYTICS_DGLR_HEAD_ALPHA"))
    print("use_drill_quality_weight =", os.getenv("ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_ENABLE"))
    print("drill_quality_refine =", os.getenv("ULTRALYTICS_DGLR_HEAD_DRILL_WEIGHT_REFINE"))
    print("dglr_target_class_ids =", os.getenv("ULTRALYTICS_DGLR_HEAD_TARGET_CLASS_IDS"))
    print("note = DGLR-Head adds glare-aware local-contrast refinement to the regression/quality branch on P3/P4.")

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
        "save_period": args.save_period,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
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
