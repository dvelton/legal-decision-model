# Policy v2 evaluation artifacts

This directory contains committed, reproducible summaries for the Northstar 2.0 policy ensemble.

| File | Contents |
|---|---|
| `benchmark-manifest.json` | Expected generated benchmark counts, seed, and split hashes |
| `test.json` | Independent-generator held-out evaluation |
| `challenge.json` | Behavioral adversarial evaluation |

The JSONL benchmark itself is generated locally under `benchmark/` and excluded from Git. Recreate
the inputs and reports with:

```bash
uv run legal-decision-model policy generate --output benchmark --seed 20261004
uv run legal-decision-model policy validate-benchmark --data benchmark
uv run legal-decision-model policy evaluate test challenge \
  --data benchmark \
  --model-dir policy-model
```

The test and challenge reports are model measurements on generated fictional data. They are not
estimates of performance on real legal requests.

| Split | False clears | Attention recall | Clearance coverage | False-clear upper bound |
|---|---:|---:|---:|---:|
| Test | 0 | 100% | 13.43% | 0.1198% |
| Challenge | 0 | 100% | 5.16% | 0.2394% |

The bounds count independent scenarios rather than correlated paraphrases. Test and challenge have
2,500 and 1,250 attention-required scenarios, so neither independently places the 95% upper bound
below the 0.1% policy target. Both observed zero false-clearance scenarios.
