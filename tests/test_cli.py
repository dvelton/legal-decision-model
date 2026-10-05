from legal_decision_model.cli import _parser
from legal_decision_model.constants import BENCHMARK_DIR, DATA_DIR, POLICY_BENCHMARK_SEED


def test_legacy_commands_keep_v1_defaults() -> None:
    parser = _parser()
    generate = parser.parse_args(["generate"])
    validate = parser.parse_args(["validate"])
    assert generate.output == str(DATA_DIR)
    assert generate.seed == 20261002
    assert validate.data == str(DATA_DIR)


def test_policy_commands_use_v2_defaults() -> None:
    parser = _parser()
    generate = parser.parse_args(["policy", "generate"])
    validate = parser.parse_args(["policy", "validate-benchmark"])
    assert generate.output == str(BENCHMARK_DIR)
    assert generate.seed == POLICY_BENCHMARK_SEED
    assert validate.data == str(BENCHMARK_DIR)
