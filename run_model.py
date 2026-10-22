"""Run a model over the test set through Ollama and save raw outputs + latency.

  python run_model.py --model llama3.2:latest --prompt baseline --out preds/baseline.jsonl
  python run_model.py --model req-classifier  --prompt short    --out preds/finetuned.jsonl
"""
import argparse, json, time
import requests
from common import SYSTEM_PROMPT, build_user_message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=["baseline", "short"], required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    template = open("prompts/baseline_prompt.txt", encoding="utf-8").read()  # exact production prompt
    test = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    if a.limit:
        test = test[:a.limit]
    with open(a.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(test, 1):
            if a.prompt == "baseline":
                # mirrors the n8n node exactly: /api/generate, single prompt, temperature 0, seed 42
                prompt = (template.replace("{{subject}}", rec["subject"]).replace("{{channel}}", "Web Form")
                          .replace("{{request_text}}", rec["message"]))
                url = f"{a.host}/api/generate"
                body = {"model": a.model, "prompt": prompt, "stream": False, "format": "json",
                        "options": {"temperature": 0, "seed": 42}}
            else:
                url = f"{a.host}/api/chat"
                body = {"model": a.model, "stream": False, "format": "json",
                        "options": {"temperature": 0, "seed": 0},
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": build_user_message(rec["subject"], rec["message"])}]}
            t0 = time.time()
            try:
                r = requests.post(url, json=body, timeout=600)
                r.raise_for_status()
                j = r.json()
                text = j["response"] if a.prompt == "baseline" else j["message"]["content"]
                out = {"id": rec["id"], "raw": text,
                       "latency_s": time.time() - t0, "prompt_tokens": j.get("prompt_eval_count")}
            except Exception as e:
                out = {"id": rec["id"], "raw": "", "error": str(e), "latency_s": time.time() - t0}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(test)}")
    print("saved", a.out)


if __name__ == "__main__":
    main()