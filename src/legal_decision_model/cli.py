"""Command-line interface."""

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from legal_decision_model.benchmark import generate_benchmark
from legal_decision_model.constants import (
    BENCHMARK_DIR,
    DATA_DIR,
    DEFAULT_POLICY_PATH,
    LEGACY_SPLIT_SIZES,
    MODEL_DIR,
    POLICY_BENCHMARK_SEED,
    POLICY_MODEL_DIR,
    POLICY_TRAINING_SEED,
)
from legal_decision_model.contribution import validate_contribution
from legal_decision_model.evaluate import evaluate_split
from legal_decision_model.generate import generate_dataset
from legal_decision_model.inference import LegalDecisionModel
from legal_decision_model.policy_evaluation import evaluate_policy_split
from legal_decision_model.policy_inference import PolicyDecisionModel
from legal_decision_model.policy_schema import load_policy
from legal_decision_model.policy_training import train_policy_model
from legal_decision_model.public_scan import scan_public_content
from legal_decision_model.train import train_model
from legal_decision_model.validate import validate_dataset
from legal_decision_model.web import create_app


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _generate(args: argparse.Namespace) -> int:
    paths = generate_dataset(Path(args.output), args.seed)
    report = validate_dataset(
        Path(args.output),
        LEGACY_SPLIT_SIZES,
        require_v2_metadata=False,
    )
    _json({"paths": {key: str(value) for key, value in paths.items()}, "report": report.__dict__})
    return 0


def _validate(args: argparse.Namespace) -> int:
    _json(
        validate_dataset(
            Path(args.data),
            LEGACY_SPLIT_SIZES,
            require_v2_metadata=False,
        ).__dict__
    )
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


def _policy_validate(args: argparse.Namespace) -> int:
    policy = load_policy(Path(args.file))
    _json(
        {
            "name": policy.name,
            "version": policy.version,
            "facts": policy.fact_names,
            "triggers": policy.triggers,
            "risk_control": policy.risk_control.__dict__,
            "model": policy.model.__dict__,
        }
    )
    return 0


def _policy_generate(args: argparse.Namespace) -> int:
    paths = generate_benchmark(Path(args.output), args.seed)
    report = validate_dataset(Path(args.output))
    _json({"paths": {key: str(value) for key, value in paths.items()}, "report": report.__dict__})
    return 0


def _policy_validate_benchmark(args: argparse.Namespace) -> int:
    _json(validate_dataset(Path(args.data)).__dict__)
    return 0


def _policy_train(args: argparse.Namespace) -> int:
    summary = train_policy_model(
        Path(args.data),
        Path(args.model_dir),
        device=args.device,
        encoder_batch_size=args.encoder_batch_size,
        seed=args.seed,
    )
    _json(summary.metadata.to_dict())
    return 0


def _policy_evaluate(args: argparse.Namespace) -> int:
    reports = {
        split: evaluate_policy_split(
            split,
            Path(args.data),
            Path(args.model_dir),
            device=args.device,
            batch_size=args.batch_size,
        ).to_dict()
        for split in args.splits
    }
    _json(reports)
    return 0


def _policy_decide(args: argparse.Namespace) -> int:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        raise ValueError("provide --text or --file")
    model = PolicyDecisionModel(Path(args.model_dir), device=args.device)
    _json(model.predict(text).to_dict())
    return 0


def _contribution_validate(args: argparse.Namespace) -> int:
    _json(validate_contribution(Path(args.file)).__dict__)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-decision-model",
        description="Synthetic proof of concept for legal-attention routing.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate the preserved v0.1 dataset")
    generate.add_argument("--output", default=str(DATA_DIR))
    generate.add_argument("--seed", type=int, default=20261002)
    generate.set_defaults(handler=_generate)

    validate = commands.add_parser("validate", help="validate the preserved v0.1 dataset")
    validate.add_argument("--data", default=str(DATA_DIR))
    validate.add_argument(
        "--legacy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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

    policy = commands.add_parser("policy", help="validate and inspect a policy file")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    policy_validate = policy_commands.add_parser("validate", help="validate a policy YAML file")
    policy_validate.add_argument("--file", default=str(DEFAULT_POLICY_PATH))
    policy_validate.set_defaults(handler=_policy_validate)

    policy_generate = policy_commands.add_parser(
        "generate",
        help="generate the source-separated policy benchmark",
    )
    policy_generate.add_argument("--output", default=str(BENCHMARK_DIR))
    policy_generate.add_argument("--seed", type=int, default=POLICY_BENCHMARK_SEED)
    policy_generate.set_defaults(handler=_policy_generate)

    policy_validate_benchmark = policy_commands.add_parser(
        "validate-benchmark",
        help="validate the source-separated policy benchmark",
    )
    policy_validate_benchmark.add_argument("--data", default=str(BENCHMARK_DIR))
    policy_validate_benchmark.set_defaults(handler=_policy_validate_benchmark)

    policy_train = policy_commands.add_parser(
        "train",
        help="train the multi-task Laya policy ensemble",
    )
    policy_train.add_argument("--data", default=str(BENCHMARK_DIR))
    policy_train.add_argument("--model-dir", default=str(POLICY_MODEL_DIR))
    policy_train.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="mps",
    )
    policy_train.add_argument("--encoder-batch-size", type=int, default=32)
    policy_train.add_argument("--seed", type=int, default=POLICY_TRAINING_SEED)
    policy_train.set_defaults(handler=_policy_train)

    policy_evaluate = policy_commands.add_parser(
        "evaluate",
        help="evaluate the multi-task policy ensemble",
    )
    policy_evaluate.add_argument("splits", nargs="*", default=["test", "challenge"])
    policy_evaluate.add_argument("--data", default=str(BENCHMARK_DIR))
    policy_evaluate.add_argument("--model-dir", default=str(POLICY_MODEL_DIR))
    policy_evaluate.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    policy_evaluate.add_argument("--batch-size", type=int, default=32)
    policy_evaluate.set_defaults(handler=_policy_evaluate)

    policy_decide = policy_commands.add_parser(
        "decide",
        help="make one policy-ensemble decision",
    )
    policy_source = policy_decide.add_mutually_exclusive_group(required=True)
    policy_source.add_argument("--text")
    policy_source.add_argument("--file")
    policy_decide.add_argument("--model-dir", default=str(POLICY_MODEL_DIR))
    policy_decide.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    policy_decide.set_defaults(handler=_policy_decide)

    contribution = commands.add_parser(
        "contribution",
        help="validate fictional behavioral test contributions",
    )
    contribution_commands = contribution.add_subparsers(
        dest="contribution_command",
        required=True,
    )
    contribution_validate = contribution_commands.add_parser(
        "validate",
        help="validate one fictional contribution JSON file",
    )
    contribution_validate.add_argument("file")
    contribution_validate.set_defaults(handler=_contribution_validate)
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
