"""Gradio demo for Hugging Face Spaces (CPU). Runs the quantized GGUF with llama-cpp-python.
Set GGUF_REPO / GGUF_FILE below to your Hugging Face model repo after you upload the GGUF.
"""
import json, re
import gradio as gr
from llama_cpp import Llama

GGUF_REPO = "Sanved13/Llama-3.2-3B-request-classifier"   # <-- change YOUR-HF-USERNAME
GGUF_FILE = "req-classifier.Q4_K_M.gguf"                     # <-- change if different
THRESHOLD = 0.70
SYSTEM = ("You are a customer-request classifier. Read the request and reply with ONLY a JSON object "
          "with keys: request_type, urgency, sub_topic, confidence, summary, customer_sentiment, "
          "entities (invoice_number, amount, currency), required_action.")

llm = Llama.from_pretrained(repo_id=GGUF_REPO, filename=GGUF_FILE, n_ctx=2048, verbose=False)


def classify(subject, message):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Subject: {subject.strip()}\nMessage: {message.strip()}"}]
    raw = llm.create_chat_completion(messages=msgs, temperature=0, max_tokens=320)["choices"][0]["message"]["content"]
    try:
        obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        conf = float(obj["confidence"])
    except Exception:
        return "INVALID OUTPUT -> human review", {"raw_output": raw}
    if obj.get("request_type") == "urgent_escalation" or obj.get("urgency") == "critical":
        route = "URGENT ESCALATION: HUMAN REVIEW REQUIRED"
    elif conf < THRESHOLD:
        route = f"LOW CONFIDENCE ({conf:.2f} < {THRESHOLD}) -> human review"
    else:
        route = f"Auto-route: {obj.get('request_type')} (confidence {conf:.2f})"
    return route, obj


examples = [
    ["Wrong invoice amount", "I was charged $450 on invoice INV-20931 but my plan is $320 per month. Please fix this."],
    ["Account security issue", "Someone changed my password and I see transfers I didn't make. Please help right now."],
    ["Support hours", "What are your customer support hours on weekends?"],
    ["Help", "I need help with something."],
]
demo = gr.Interface(
    fn=classify,
    inputs=[gr.Textbox(label="Subject"), gr.Textbox(label="Message", lines=5)],
    outputs=[gr.Textbox(label="Routing decision"), gr.JSON(label="Model output")],
    examples=examples, title="Fine-tuned customer-request classifier",
    description="Llama 3.2 3B + LoRA. The model classifies; routing rules are deterministic code. "
                "Running on a free CPU, so allow ~20-40 seconds per request.")
if __name__ == "__main__":
    demo.launch()