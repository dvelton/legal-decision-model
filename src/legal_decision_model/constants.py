"""Shared project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "model"

NO_ATTENTION = "NO_HUMAN_LAWYER_ATTENTION"
REQUIRES_ATTENTION = "REQUIRES_HUMAN_LAWYER_ATTENTION"
LABELS = (NO_ATTENTION, REQUIRES_ATTENTION)
SITE_NAME = "requires_human_lawyer_attention"
POLICY_VERSION = "northstar-1.0"
BASE_MODEL = "convaiinnovations/laya"
BASE_MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

SPLIT_SIZES = {
    "train": 7500,
    "validation": 1000,
    "test": 1000,
    "challenge": 500,
}
