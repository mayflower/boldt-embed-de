# AutoResearch Run: baseline

Status: ok
Mode: real
Budget: 20 minutes
Elapsed: 32.527 seconds
Deadline respected: yes
Invalid for default loop: no

## Command

```bash
python scripts/ar_run_trial.py --config configs/autoresearch/experiments/current.json --out outputs/autoresearch/baseline --real --allow-gpu --notes baseline = dense-v6.1 eval under the AR harness
```

## Git

- commit: 1c8744386b809766e75f5e4c6a219592e50fd5e6
- dirty: True
- diffstat saved: git.diffstat

## Config

- config path: configs/autoresearch/experiments/current.json
- resolved config: config.resolved.json

## Metrics

| metric | value |
|---|---:|
| WebFAQ Recall@100 | 0.9765 |
| WebFAQ nDCG@10 | 0.7044 |
| GermanQuAD nDCG@10 | 0.878 |
| DT-test nDCG@10 | 0.9748 |
| Matryoshka 256 retention | 1.0007 |

## Notes

baseline = dense-v6.1 eval under the AR harness
