import argparse

from ultralytics import YOLO

from train_yolov8s_afpn_atfl_inneriou_dlq_head import (
    DATA_YAML,
    DEFAULT_DEVICE,
    DEFAULT_INNER_RATIO,
    MODEL_CFG,
    PRETRAINED_WEIGHTS,
    configure_env,
    str2bool,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8s AFPN with ATFL + Inner-IoU + DLQ balanced weighting and focused TAL."
        )
    )
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset yaml path.")
    parser.add_argument("--model", default=MODEL_CFG, help="Model yaml or checkpoint path.")
    parser.add_argument("--pretrained", default=PRETRAINED_WEIGHTS, help="Pretrained weights for yaml models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument(
        "--name",
        default="yolov8s_afpn_atfl_inneriou_dlq_balanced_ftal_3cls_seed42_e300",
        help="Experiment name.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer, e.g. SGD or AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=300, help="Early stopping patience.")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every x epochs.")
    parser.add_argument("--cos-lr", type=str2bool, default=True, help="Use cosine learning rate schedule.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=30,
        help="Disable mosaic augmentation in the last N epochs.",
    )
    parser.add_argument("--inner-ratio", type=float, default=DEFAULT_INNER_RATIO, help="Inner-IoU box scaling ratio.")
    parser.add_argument(
        "--dlq-levels",
        default="p3p4",
        help="DLQ head levels: 'p3p4' (recommended) or 'all'.",
    )
    parser.add_argument("--dlq-lambda", type=float, default=0.15, help="Weight of the DLQ quality loss term.")
    parser.add_argument(
        "--dlq-base-level",
        type=int,
        default=3,
        help="Pyramid base level used to parse dlq-levels for the current detect head.",
    )
    parser.add_argument(
        "--dlq-score-mode",
        default="mul",
        choices=["mul", "pow"],
        help="How cls and quality scores are fused during inference.",
    )
    parser.add_argument(
        "--dlq-alpha",
        type=float,
        default=0.6,
        help="Alpha for score fusion when dlq-score-mode=pow.",
    )
    parser.add_argument(
        "--use-drill-quality-weight",
        type=str2bool,
        default=True,
        help="Enable balanced drill_pipe-aware weighting for the DLQ quality loss.",
    )
    parser.add_argument(
        "--drill-quality-refine",
        type=str2bool,
        default=True,
        help="Enable refined balanced weighting for very small drill_pipe targets.",
    )
    parser.add_argument(
        "--dlq-target-class-ids",
        default="2",
        help="Comma-separated target class ids used by drill-aware DLQ weighting.",
    )
    parser.add_argument("--drill-quality-base-weight", type=float, default=1.15, help="Flat drill_pipe DLQ weight.")
    parser.add_argument("--drill-quality-small-h1", type=float, default=0.06, help="Small-target height threshold.")
    parser.add_argument("--drill-quality-small-h2", type=float, default=0.09, help="Reserved secondary height threshold.")
    parser.add_argument("--drill-quality-small-w1", type=float, default=1.25, help="Weight for very small drill_pipe.")
    parser.add_argument("--drill-quality-small-w2", type=float, default=1.10, help="Reserved secondary drill weight.")
    parser.add_argument(
        "--focused-tal-enable",
        type=str2bool,
        default=True,
        help="Enable focused Task-Aligned Assigner for target classes.",
    )
    parser.add_argument(
        "--focused-tal-topk",
        type=int,
        default=24,
        help="Top-k candidates used by focused TAL.",
    )
    parser.add_argument(
        "--focused-tal-alpha",
        type=float,
        default=1.0,
        help="Classification exponent in focused TAL alignment metric.",
    )
    parser.add_argument(
        "--focused-tal-beta",
        type=float,
        default=2.9,
        help="Localization exponent in focused TAL alignment metric.",
    )
    parser.add_argument(
        "--focused-tal-boost",
        type=float,
        default=2.0,
        help="Alignment boost applied to target classes in focused TAL.",
    )
    parser.add_argument(
        "--focused-tal-target-class-ids",
        default="2",
        help="Comma-separated target class ids used by focused TAL boost.",
    )
    parser.add_argument("--cache", type=str2bool, default=False, help="Whether to cache the dataset.")
    parser.add_argument("--amp", type=str2bool, default=True, help="Whether to use AMP training.")
    parser.add_argument("--deterministic", type=str2bool, default=True, help="Whether to use deterministic training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the provided checkpoint path.")
    return parser.parse_args()


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
    print("cls_loss =", "atfl")
    print("iou_loss =", "inner_iou")
    print("inner_ratio =", args.inner_ratio)
    print("dlq_head_enable =", 1)
    print("dlq_head_levels =", args.dlq_levels)
    print("dlq_lambda =", args.dlq_lambda)
    print("dlq_score_mode =", args.dlq_score_mode)
    print("dlq_alpha =", args.dlq_alpha)
    print("dlq_base_level =", args.dlq_base_level)
    print("use_drill_quality_weight =", int(args.use_drill_quality_weight))
    print("drill_quality_refine =", int(args.drill_quality_refine))
    print("dlq_target_class_ids =", args.dlq_target_class_ids)
    print("focused_tal_enable =", int(args.focused_tal_enable))
    print("focused_tal_topk =", args.focused_tal_topk)
    print("focused_tal_alpha =", args.focused_tal_alpha)
    print("focused_tal_beta =", args.focused_tal_beta)
    print("focused_tal_boost =", args.focused_tal_boost)
    print("focused_tal_target_class_ids =", args.focused_tal_target_class_ids)
    print("note = YOLOv8s AFPN + ATFL + Inner-IoU + DLQ balanced + focused TAL")

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
