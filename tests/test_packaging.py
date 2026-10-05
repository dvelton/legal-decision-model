from legal_decision_model.constants import PACKAGED_POLICY_PATH, SOURCE_POLICY_PATH
from legal_decision_model.policy_schema import load_policy


def test_packaged_policy_matches_authoring_source() -> None:
    assert PACKAGED_POLICY_PATH.read_bytes() == SOURCE_POLICY_PATH.read_bytes()
    assert load_policy(PACKAGED_POLICY_PATH) == load_policy(SOURCE_POLICY_PATH)
