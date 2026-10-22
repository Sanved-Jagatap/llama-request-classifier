"""Independent label check: a second local model classifies each CLEAR synthetic message WITHOUT seeing
our intended label. build_dataset.py then drops rows where the two disagree.

  ollama pull llama3:8b
  python verify_labels.py                       # resumable; writes data/raw/verdicts.jsonl
  python build_dataset.py                       # uses the verdicts automatically
"""
import argparse, hashlib, json, os, threading
from concurrent.futures import ThreadPoolExecutor
import requests
from common import REQUEST_TYPES, build_user_message

RUBRIC = """Classify the customer message into exactly ONE type.

- complaint: the customer is dissatisfied or disputes something that already happened (wrong or unexpected charge, billing discrepancy, slow or poor service, broken product, unresolved ticket). Anger or the word "ASAP" does NOT make it urgent_escalation.
- general_enquiry: the customer only asks for information (hours, pricing, policies, how a feature works) and wants nothing changed or fixed.
- service_request: the customer asks the company to DO a routine account action (change email/phone/address/company name, add or remove seats, cancel, upgrade/downgrade, send an invoice copy, reset 2FA or a password with no sign of compromise).
- urgent_escalation: a security or access emergency or severe business impact that is happening now (account hacked or compromised, unauthorized transactions, locked out of a critical/admin account, data breach, leaked credentials, outage affecting customers).

Reply with ONLY JSON: {"request_type": "<complaint|general_enquiry|service_request|urgent_escalation>"}"""


def key_of(msg):
    return hashlib.sha1(msg.strip().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/raw.jsonl")
    ap.add_argument("--out", default="data/raw/verdicts.jsonl")
    ap.add_argument("--model", default="llama3:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.raw, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("clarity") == "clear"]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if key_of(r["message"]) not in done]
    print(f"{len(rows)} clear rows, {len(done)} already verified, {len(todo)} to do")
    lock, n = threading.Lock(), {"i": 0}

    def work(r):
        body = {"model": a.model, "stream": False, "format": "json", "options": {"temperature": 0, "seed": 0},
                "messages": [{"role": "system", "content": RUBRIC},
                             {"role": "user", "content": build_user_message(r["subject"], r["message"])}]}
        try:
            resp = requests.post(f"{a.host}/api/chat", json=body, timeout=300)
            resp.raise_for_status()
            t = json.loads(resp.json()["message"]["content"]).get("request_type")
        except Exception as e:
            t = None
        with lock:
            if t in REQUEST_TYPES:
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key_of(r["message"]), "verifier_type": t, "model": a.model}) + "\n")
            n["i"] += 1
            if n["i"] % 50 == 0:
                print(f"{n['i']}/{len(todo)}")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print("done ->", a.out)


if __name__ == "__main__":
    main()