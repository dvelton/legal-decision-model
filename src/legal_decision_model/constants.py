"""Shared project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "model"
POLICIES_DIR = PROJECT_ROOT / "policies"
SOURCE_POLICY_PATH = POLICIES_DIR / "northstar.yaml"
PACKAGED_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "northstar.yaml"
DEFAULT_POLICY_PATH = SOURCE_POLICY_PATH if SOURCE_POLICY_PATH.is_file() else PACKAGED_POLICY_PATH
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
POLICY_MODEL_DIR = PROJECT_ROOT / "policy-model"

NO_ATTENTION = "NO_HUMAN_LAWYER_ATTENTION"
REQUIRES_ATTENTION = "REQUIRES_HUMAN_LAWYER_ATTENTION"
LABELS = (NO_ATTENTION, REQUIRES_ATTENTION)
SITE_NAME = "requires_human_lawyer_attention"
POLICY_VERSION = "northstar-2.0"
LEGACY_POLICY_VERSION = "northstar-1.0"
POLICY_BENCHMARK_SEED = 20261004
POLICY_TRAINING_SEED = 20261004
BASE_MODEL = "convaiinnovations/laya"
BASE_MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

SPLIT_SIZES = {
    "train": 75000,
    "validation": 12000,
    "test": 10000,
    "challenge": 5000,
}

LEGACY_SPLIT_SIZES = {
    "train": 7500,
    "validation": 1000,
    "test": 1000,
    "challenge": 500,
}
