---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags: [text-classification, lora, qlora, customer-support, gguf]
---
# Llama-3.2-3B-request-classifier

**Built with Llama.** This model is a fine-tune of Llama 3.2 3B Instruct and is subject to the Llama 3.2 Community License.

Classifies customer requests into `complaint | general_enquiry | service_request | urgent_escalation`, with urgency, sentiment, entities and a clarity-based confidence, returning strict JSON.

## Files in this repo
| File | What it is |
|---|---|
| `req-classifier.Q4_K_M.gguf` | Quantized model (about 2 GB) for Ollama / llama.cpp |
| `adapter/` | The LoRA adapter (about 110 MB) for use with the base model |
| `Modelfile` | Ollama template, system prompt and parameters |

## Quick start (Ollama)
```bash
# put the .gguf and Modelfile in one folder, then:
ollama create req-classifier -f Modelfile
ollama run req-classifier "Subject: Test
Message: I was charged twice for invoice INV-1234"
```
Input format is exactly `Subject: ...` then `Message: ...`; the system prompt is built into the Modelfile.

## Intended use
Advisory triage inside a workflow where deterministic code makes all routing decisions and low-confidence/invalid outputs go to a human. **Not** intended for autonomous decisions.

## Training data
- 1,617 training and 202 validation requests, synthetic, generated with mistral:7b (local), with type/urgency/clarity set by scenario design, validated by rules (no invented entities, urgency/type consistency), near-duplicates removed, and filtered by agreement with an independent model (llama3:8b).
- Test set: 288 items, including 86 hand-written examples never used in training.
- Manual review of 80 random training rows: 6.2% label errors (reviewed with AI assistance).

## Training
QLoRA (4-bit), rank 16, alpha 16, 3 epochs, LR 2e-4, loss on assistant tokens only. Hardware: Google Colab T4 (free tier). Training time: <fill in from Colab>.

## Results (288-item test set; GGUF Q4_K_M run through Ollama; baseline = prompt-only llama3.2)
| Metric | Baseline | This model |
|---|---|---|
| Type accuracy (clear items) | 78.3% | 98.1% |
| Hand-written subset type accuracy (73) | 83.6% | 97.3% |
| Emergency recall | 97.0% | 98.5% |
| False-emergency rate | 10.2% | 0.0% |
| Hallucinated entities | 13.2% | 3.1% |
| Full-schema compliance | 71.9% | 99.7% |
| Overall human-review rate | 16.7% | 5.6% |
| Prompt tokens / latency (mean) | 998 / 2.64 s | 128 / 1.51 s |

Urgency accuracy rose from 51.7% to 69.6%, but urgency labels are partly arbitrary; treat that figure cautiously.

## Limitations
- Mostly synthetic data; real customer language will differ.
- `confidence` reflects request clarity (trained from clarity tiers), not a calibrated probability of correctness.
- English only. Small test set; confidence intervals are wide.
- Can miss non-security emergencies (outages, legal threats).
- Not evaluated for fairness across dialects or languages.