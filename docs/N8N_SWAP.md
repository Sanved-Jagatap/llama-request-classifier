# Swapping the fine-tuned model into the n8n workflow (v6)

Only TWO nodes change. Everything after classification (Parse & Validate, 0.70 confidence gate, branches, audit log) stays as it is, because the fine-tuned model returns the same JSON and we keep using `/api/generate`.

## 1. Register the model in Ollama (PowerShell, in the project folder)
```powershell
ollama create req-classifier -f export\Modelfile
ollama run req-classifier "Subject: Test`nMessage: I was charged twice for invoice INV-1234"
```
You should get a JSON object back.

## 2. Edit the "Build Classification Prompt" Code node
Duplicate the workflow first (menu > Duplicate) so you keep the original for comparison. Then replace the whole code with:
```javascript
const item = $input.first().json;

// Fine-tuned model: the instructions live in the model (Modelfile SYSTEM line),
// so the prompt is only the request itself, in the same format used for training.
const prompt = `Subject: ${item.subject}\nMessage: ${item.request_text}`;

return [{ json: { ...item, classification_prompt: prompt } }];
```

## 3. Edit the "Classify Request - Ollama" HTTP Request node
Change only the model name in the JSON body expression:
```
={{ JSON.stringify({ model: "req-classifier", prompt: $json.classification_prompt, stream: false, format: "json", options: { temperature: 0, seed: 42 } }) }}
```
(The URL stays `http://host.docker.internal:11434/api/generate`, and the Parse node keeps reading `httpResult.response`.)

## 4. Test
Re-run your 5 original test cases from the form and compare with the original workflow: classification, confidence, branch taken. The vague message ("I need help with something.") should still go to Human Review. Screen-record the run for the portfolio.

Tip: keep both versions published/saved (original = baseline, duplicate = fine-tuned) so you can show them side by side.