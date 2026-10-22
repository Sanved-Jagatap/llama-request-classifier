---
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - text-classification
  - lora
  - qlora
  - customer-support
  - gguf
---

# Llama-3.2-3B-request-classifier

**Built with Llama.**

Llama-3.2-3B-request-classifier is a fine-tuned version of Llama 3.2 3B Instruct designed to classify customer requests into four categories:

- `complaint`
- `general_enquiry`
- `service_request`
- `urgent_escalation`

The model also predicts urgency, sentiment, sub-topic, entities, a clarity-based confidence score, a summary, and a required action. Outputs are intended to follow a structured JSON schema.

This model is based on Llama 3.2 3B Instruct and is subject to the Llama 3.2 Community License.

## Files

| File | Description |
|---|---|
| `req-classifier.Q4_K_M.gguf` | Quantized model, approximately 2 GB, for compatible Ollama and llama.cpp workflows |
| `adapter/` | LoRA adapter, approximately 110 MB, for use with the compatible base model |
| `Modelfile` | Ollama configuration containing the model reference, system prompt, and inference parameters |

File names, availability, and paths depend on the artifacts published in the model repository. Verify that each artifact is available before using the corresponding instructions.

## Quick start with Ollama

Place the GGUF model file and `Modelfile` in the same directory. Ensure the model-file path in `Modelfile` matches the actual GGUF filename and location.

Run:

```bash
ollama create req-classifier -f Modelfile
ollama run req-classifier
```

Example input:

```text
Subject: Test
Message: I was charged twice for invoice INV-1234
```

The system prompt and model configuration are defined in `Modelfile`. These instructions assume that Ollama is installed and running and that the required model file is available locally.

## Intended use

This model is intended for advisory customer-request triage in workflows where deterministic application code makes routing decisions.

Potential applications include:

- Customer-support request classification.
- Preliminary urgency assessment.
- Structured extraction of request details.
- Suggested actions and summaries for support teams.
- Integration into automated workflows with validation and human oversight.

Low-confidence, invalid, ambiguous, or otherwise uncertain outputs should be escalated to a human reviewer.

The model is not intended for autonomous high-impact decisions. Predictions must not be treated as guaranteed correct, and model-generated urgency labels must not replace appropriate operational escalation procedures.

## Training data

- **Training set:** 1,617 synthetic customer requests.
- **Validation set:** 202 synthetic customer requests.
- **Data generation:** Generated locally using `mistral:7b`. Request types, urgency, and clarity were assigned through scenario design and validation rules.
- **Quality checks:** Validation rules checked entity consistency and type/urgency consistency. Near-duplicates were removed, and examples were filtered based on agreement with an independent `llama3:8b` model.
- **Test set:** 288 examples, including 86 AI-assisted examples reviewed by the author. These examples were not used for training.
- **Manual review:** A sample of 80 training rows was reviewed with AI assistance; 6.2% contained label errors.

The test examples were drafted with AI assistance and reviewed by the author. They are not real customer traffic and may share labeling conventions with the training data. Consequently, these results should be interpreted as internal evaluation results rather than evidence of production performance.

The manual-review result applies only to the reviewed sample; it should not be interpreted as a verified error rate for the entire dataset.

## Training configuration

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-3B-Instruct` |
| Fine-tuning method | QLoRA with 4-bit model loading |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Epochs | 3 |
| Learning rate | `2e-4` |
| Training examples | 1,617 |
| Validation examples | 202 |
| Hardware | Google Colab NVIDIA Tesla T4 GPU |
| Observed training-cell runtime | Approximately 26 minutes and 36 seconds |
| Final training loss | `0.202253` |
| Final validation loss | `0.264688` |
| Loss masking | Training loss was calculated on assistant response tokens only |

The reported runtime refers to the training cell observed in the Colab run. It does not necessarily represent the complete time required for environment setup, model loading, dataset preparation, evaluation, or GGUF export.

## Evaluation

The project reports results from a 288-example test set. Metrics below come from separate evaluation setups and should not be combined into a single performance estimate without verifying that the test subsets, metric definitions, and evaluation conditions are equivalent.

### GGUF Q4_K_M through Ollama

The following table reports the project's comparison between a prompt-only `llama3.2` baseline and the fine-tuned GGUF Q4_K_M model running through Ollama.

| Metric | Baseline | Fine-tuned model |
|---|---:|---:|
| Type accuracy on clear items | 78.3% | 98.1% |
| AI-assisted, author-reviewed clear-subset type accuracy (73 examples) | 83.6% | 97.3% |
| Emergency recall | 97.0% | 98.5% |
| False-emergency rate | 10.2% | 0.0% |
| Hallucinated entities | 13.2% | 3.1% |
| Full-schema compliance | 71.9% | 99.7% |
| Overall human-review rate | 16.7% | 5.6% |
| Mean prompt tokens | 998 | 128 |
| Mean latency | 2.64 s | 1.51 s |

Urgency accuracy increased from 51.7% to 69.6% in the reported comparison. Urgency remains a weaker prediction dimension, and the labels partly depend on scenario design. This result should therefore be interpreted cautiously.

**Interpretation caveat:** This comparison reflects the evaluated configurations, including their prompting and inference settings. Prompt length differs substantially between the baseline and fine-tuned configurations. Therefore, the observed differences cannot be attributed exclusively to fine-tuning without a controlled comparison that holds the other relevant conditions constant.

Metric denominators and definitions should be consulted in the evaluation scripts and result files before interpreting these percentages as directly comparable measurements. In particular, emergency recall and false-emergency rate measure different aspects of performance.

The reported emergency-recall change is modest in absolute terms. If the underlying counts are 64/66 versus 65/66, it represents one additional detected emergency, rather than a large improvement. The underlying evaluation counts should be verified before using this interpretation in external reporting.

### Separate Colab evaluation

The training notebook also reports the following results for a test-set evaluation:

| Metric | Reported result |
|---|---:|
| Test examples | 288 |
| Type accuracy | 96.58% |
| Type macro-F1 | 96.19% |
| Emergency recall | 98.48% |
| False-emergency rate | 0.0% |

These Colab results are reported separately from the Ollama/GGUF comparison. The inference setup, evaluation procedure, and subset definitions should be checked before making direct comparisons between the figures.

The reported results describe performance on the project's evaluation data. They do not establish equivalent performance on real customer traffic or guarantee detection of every urgent request.

## Limitations

- **Synthetic data:** Most training and evaluation examples are synthetic. Real customer messages may differ in wording, context, ambiguity, and distribution.
- **Potential shared labeling conventions:** AI-assisted test examples were reviewed by the author but may still reflect conventions shared with the training data.
- **Limited manual review:** The reported 6.2% label-error rate comes from a sample of 80 training rows and may not represent the entire dataset.
- **Urgency prediction:** Urgency accuracy is weaker than request-type classification accuracy.
- **Confidence interpretation:** The `confidence` field represents request clarity as learned from clarity tiers. It is not a calibrated probability that the model's prediction is correct.
- **Limited evaluation size:** The test set is relatively small, so performance estimates have uncertainty and may not generalize to production.
- **Emergency misses:** The model may miss important non-security emergencies, including service outages and legal threats. A zero false-emergency rate on a test set does not establish that the model is safe for autonomous triage.
- **Language coverage:** The project was developed and evaluated primarily in English.
- **Fairness:** The model has not been evaluated for fairness across dialects or demographic groups.
- **Quantization and runtime differences:** Results from the quantized GGUF model running through Ollama may differ from results obtained in Colab because the exported model and inference setup differ.
- **Evaluation comparability:** The reported evaluations should not be treated as interchangeable unless their subsets, metric definitions, prompts, and inference conditions have been verified.
- **Human oversight:** Invalid, uncertain, ambiguous, and high-impact cases require deterministic safeguards and human review.

## Recommended safeguards

For integrations into automated workflows:

1. Validate the model output against the expected JSON schema.
2. Check predicted categories against an explicit allowlist.
3. Apply deterministic application rules for routing and escalation.
4. Route invalid, ambiguous, or uncertain outputs to human review.
5. Avoid relying on the model's confidence field as a calibrated correctness probability.
6. Log predictions and routing outcomes for subsequent evaluation.
7. Test the system against representative real-world examples before considering production deployment.

These safeguards should be implemented in the surrounding application rather than assumed to be guaranteed by the model itself.

## License

This model is based on Llama 3.2 3B Instruct and is subject to the **Llama 3.2 Community License**. Users should review the applicable license and usage conditions before using or redistributing the model.

The surrounding project code is separately licensed under the MIT License; see the repository's LICENSE file. This does not replace or override the Llama 3.2 Community License governing the base model and its applicable use and redistribution conditions.
