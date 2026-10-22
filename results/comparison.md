| Metric | baseline | finetuned |
|---|---|---|
| Accepted by workflow's own parser | 100.0% | 99.7% |
| Full schema compliance | 71.9% | 99.7% |
| Type accuracy (clear) | 78.3% | 98.1% |
| Type macro-F1 (clear) | 77.9% | 98.2% |
| Urgency accuracy (clear) | 51.7% | 69.6% |
| Urgency macro-F1 (clear) | 49.1% | 67.1% |
| Emergency recall | 97.0% | 98.5% |
| False-emergency rate (lower better) | 10.2% | 0.0% |
| HAND-WRITTEN subset: type accuracy | 83.6% | 97.3% |
| HAND-WRITTEN subset: emergency recall | 89.5% | 94.7% |
| HAND-WRITTEN subset: ambiguity recall | 76.9% | 92.3% |
| Entity: invoice match | 89.3% | 95.3% |
| Entity: amount match | 79.5% | 99.1% |
| Entity: currency match | 89.3% | 97.2% |
| Hallucinated entities (lower better) | 13.2% | 3.1% |
| Ambiguity recall (sent to review) | 52.0% | 60.0% |
| Ambiguity precision | 27.1% | 93.8% |
| Needless review on clear items (lower better) | 13.3% | 0.4% |
| Overall human-review rate | 16.7% | 5.6% |
| ECE (secondary, lower better) | 0.132 | 0.105 |
| Latency mean (s) | 2.64 | 1.51 |
| Latency p95 (s) | 3.07 | 1.75 |
| Prompt tokens (mean) | 998 | 128 |

Test set: 288 items (263 clear, 25 ambiguous); entity metrics over 215 items.