# Phase 1 Spec: Fine-Tuned Request Classifier

## Goal
Replace the prompt-engineered `llama3.2` classification step in the n8n request-triage workflow with a LoRA-fine-tuned small model, and prove (with numbers) whether it is better.

## Task definition
**Input:** `Subject: ...\nMessage: ...`
**Output (strict JSON, same schema the workflow already uses):**
`request_type, urgency, sub_topic, confidence, summary, customer_sentiment, entities{invoice_number, amount, currency}, required_action`

Labelling rules (enforced by `common.validate_record`):
- `urgency == "critical"` if and only if `request_type == "urgent_escalation"`.
- `amount` is a plain number string; `currency` is a 3-letter ISO code; two amounts -> the charged amount goes in `amount`.
- Entities must literally appear in the message (no invention).
- `confidence` reflects **clarity of classification**, not message length. In training data it is set from a clarity tier: clear -> 0.85-0.98, ambiguous -> 0.30-0.60 (the model learns to detect vagueness; it is not a calibrated probability of correctness).

## Data
| Item | Decision |
|---|---|
| Hand-written seed | 60-100 examples in `data/seed/handwritten.jsonl`, used as **test only** |
| Synthetic | ~3,000 raw (600 calls x 5), expect ~2,500 after validation + dedupe |
| Mix | ~22% each type, ~12% ambiguous; 15% "routine but urgent-sounding" hard negatives, 20% "calm but genuine emergency" |
| Split | 80/10/10 stratified by type x clarity; hand-written -> test; near-duplicate and leakage filter (Jaccard >= 0.8) |
| Test set rule | Never used for tuning. Look at it once, at the end. |

## Metrics (all computed by one eval script, same for baseline and fine-tuned)
| Metric | Definition |
|---|---|
| Type accuracy / macro-F1 | `request_type` vs gold, on **clear** items (ambiguous items have no reliable gold type and are scored by ambiguity detection) |
| Urgency accuracy / macro-F1 | `urgency` vs gold on clear items |
| Emergency recall / false-emergency rate | gold `urgent_escalation` caught as urgent type OR critical urgency (this is how the workflow's branch is forced); missing a real emergency is the costly error |
| Entity exact-match | invoice_number, amount, currency each exact match; plus hallucinated-entity rate |
| JSON validity rate | output parses and passes `validate_record`-style schema checks |
| Ambiguity detection | at threshold 0.70: precision/recall of "send to human review" on ambiguous items |
| Human-review rate | % of all test requests with confidence < 0.70 or invalid JSON |
| Calibration (secondary) | ECE and reliability diagram; interpret carefully, see note above |
| Latency / throughput | seconds per request on your machine, same hardware for both models |
| Prompt size | tokens in prompt (baseline long prompt vs short fine-tuned prompt) |

## Fair comparison protocol
- Baseline = current workflow prompt + `llama3.2`, `format: json`, temperature 0.
- Fine-tuned = short system prompt (`common.SYSTEM_PROMPT`), temperature 0.
- Same test set, same parsing/validation, same hardware.
- Report the baseline's invalid-JSON cases as failures, not as skipped rows.

## Definition of done for Phases 1-2
- [ ] This spec committed
- [ ] `handwritten.jsonl` has 60+ examples (include tricky pairs: urgent-sounding billing complaints vs real security issues)
- [ ] `data/train.jsonl`, `val.jsonl`, `test.jsonl` built
- [ ] 10% random manual review of synthetic data done; label-error rate noted in a short data card
