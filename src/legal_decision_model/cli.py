"""Command-line interface."""

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from legal_decision_model.constants import DATA_DIR, MODEL_DIR
from legal_decision_model.evaluate import evaluate_split
from legal_decision_model.generate import generate_dataset
from legal_decision_model.inference import LegalDecisionModel
from legal_decision_model.public_scan import scan_public_content
from legal_decision_model.train import train_model
from legal_decision_model.validate import validate_dataset
from legal_decision_model.web import create_app


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _generate(args: argparse.Namespace) -> int:
    paths = generate_dataset(Path(args.output), args.seed)
    report = validate_dataset(Path(args.output))
    _json({"paths": {key: str(value) for key, value in paths.items()}, "report": report.__dict__})
    return 0


def _validate(args: argparse.Namespace) -> int:
    _json(validate_dataset(Path(args.data)).__dict__)
    return 0


def _train(args: argparse.Namespace) -> int:
    summary = train_model(
        Path(args.data),
        Path(args.model_dir),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_false_clear_rate=args.max_false_clear_rate,
    )
    _json(summary.metadata.to_dict())
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    reports = {
        split: evaluate_split(
            split,
            Path(args.data),
            Path(args.model_dir),
            device=args.device,
        ).to_dict()
        for split in args.splits
    }
    _json(reports)
    return 0


def _decide(args: argparse.Namespace) -> int:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("provide --text or --file")
    model = LegalDecisionModel(Path(args.model_dir), device=args.device)
    _json(model.predict(text).to_dict())
    return 0


def _serve(args: argparse.Namespace) -> int:
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def _scan(args: argparse.Namespace) -> int:
    findings = scan_public_content(Path(args.root))
    _json({"findings": [finding.__dict__ for finding in findings], "count": len(findings)})
    return 1 if findings else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-decision-model",
        description="Synthetic proof of concept for legal-attention routing.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate all synthetic dataset splits")
    generate.add_argument("--output", default=str(DATA_DIR))
    generate.add_argument("--seed", type=int, default=20261002)
    generate.set_defaults(handler=_generate)

    validate = commands.add_parser("validate", help="validate generated data")
    validate.add_argument("--data", default=str(DATA_DIR))
    validate.set_defaults(handler=_validate)

    train = commands.add_parser("train", help="train and calibrate the Laya decision head")
    train.add_argument("--data", default=str(DATA_DIR))
    train.add_argument("--model-dir", default=str(MODEL_DIR))
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="mps")
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=20261002)
    train.add_argument("--max-false-clear-rate", type=float, default=0.0)
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="evaluate held-out synthetic splits")
    evaluate.add_argument("splits", nargs="*", default=["test", "challenge"])
    evaluate.add_argument("--data", default=str(DATA_DIR))
    evaluate.add_argument("--model-dir", default=str(MODEL_DIR))
    evaluate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    evaluate.set_defaults(handler=_evaluate)

    decide_parser = commands.add_parser("decide", help="make one model decision")
    source = decide_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file")
    decide_parser.add_argument("--model-dir", default=str(MODEL_DIR))
    decide_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    decide_parser.set_defaults(handler=_decide)

    serve = commands.add_parser("serve", help="run the local demonstration web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)

    scan = commands.add_parser("scan-public", help="scan files for public-release concerns")
    scan.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    scan.set_defaults(handler=_scan)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the selected command."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        code = args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    raise SystemExit(code)
