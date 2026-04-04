"""Local launcher for the Ultralytics CLI.

This wrapper makes the current repository importable before any globally installed
``ultralytics`` package and then forwards all command-line arguments to the
standard Ultralytics CLI entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rewrite_resume_path_args(argv: list[str]) -> list[str]:
    """Rewrite ``resume=<ckpt>`` into ``model=<ckpt> resume=True`` when needed.

    Ultralytics 8.3.248 expects resume runs to be started as
    ``model=path/to/last.pt resume=True``. Passing ``resume=path/to/last.pt``
    directly can be overwritten by the default model resolution logic, so this
    wrapper normalizes that shorthand into the form the trainer actually uses.
    """

    args = list(argv)
    resume_idx = next((i for i, arg in enumerate(args) if arg.startswith("resume=")), None)
    if resume_idx is None:
        return args

    resume_value = args[resume_idx].split("=", 1)[1].strip()
    if not resume_value:
        return args

    if resume_value.lower() in {"true", "false"}:
        return args

    has_model_arg = any(arg.startswith("model=") for arg in args)
    if not has_model_arg:
        args.insert(resume_idx, f"model={resume_value}")
        resume_idx += 1
    args[resume_idx] = "resume=True"
    return args


def main() -> None:
    """Run the Ultralytics CLI using the current process arguments."""
    from ultralytics.cfg import entrypoint

    args = _rewrite_resume_path_args(sys.argv[1:])
    print(f"run_ultralytics_cli effective args: {args}", flush=True)
    entrypoint(debug="yolo " + " ".join(args))


if __name__ == "__main__":
    main()
