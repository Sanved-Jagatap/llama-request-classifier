"""Generate synthetic labelled customer requests.

request_type, urgency and clarity are CHOSEN by us per scenario and given to the LLM,
so labels are correct by construction; the LLM writes the message and the descriptive
fields. Confidence is assigned from the clarity tier, not by the LLM.

Usage:
  python generate_dataset.py --dry-run
  python generate_dataset.py --backend anthropic --model claude-sonnet-5-5 --calls 600 --per-call 5
  python generate_dataset.py --backend ollama --model mistral:7b --calls 600   # lower quality, free
"""
import argparse, json, os, random, threading
from concurrent.futures import ThreadPoolExecutor
from common import *

TYPE_WEIGHTS = {"complaint": .22, "general_enquiry": .22, "service_request": .22,
                "urgent_escalation": .22, "ambiguous": .12}

PROMPT = """You are generating synthetic training data for a customer-request classifier at a SaaS / IT-services company (billing, accounts, support).

Write {n} DIFFERENT customer messages for this scenario:
- request_type: {rtype}
- urgency: {urgency} (low = no time pressure, medium = wants it handled soon, high = clearly pressing, critical = security/access emergency)
- topic area: {topic}
- clarity: {clarity}
- tone: {tone}
- length: {length}
- quirk: {quirk}

Rules:
- Write as the customer. Each item must differ in wording, details and structure.
- Never use real company names. Invent plausible but fictional details.
- Put an invoice number or an amount only if it fits naturally. If you mention an invoice number, pick one from this list: {inv_hint}. Use varied, realistic amounts (not round numbers like 100 or 500). Amount in "entities" must be a plain number string (no symbols/commas), currency a 3-letter ISO code. If two amounts appear (charged vs expected), put the CHARGED amount in "amount" and mention the discrepancy in "summary". Use null for anything not present in the message. Never put in entities something that is not literally in the message.
- clarity "clear": the request is unambiguous. clarity "ambiguous": the message is genuinely vague: at most 12 words, no invoice number, no amount, no named product or problem, so a support agent could not tell what the customer wants (examples of the style: "I need help with something." / "Can you look into this?"). "sub_topic" must be "unclear".
- "sub_topic": 2-5 words. "summary": one factual sentence, nothing invented. "required_action": one short imperative sentence for the support team.
- "customer_sentiment" must be EXACTLY one of: "positive", "neutral", "negative" (it is not the same as the tone; an anxious or angry customer is "negative").
- "entities" must always be an object with exactly the three keys invoice_number, amount, currency (use null for each missing value).

Return ONLY a JSON object: {{"items": [ {{"subject": str, "message": str, "sub_topic": str, "summary": str, "customer_sentiment": str, "entities": {{"invoice_number": str|null, "amount": str|null, "currency": str|null}}, "required_action": str}} ]}} with exactly {n} items."""


def make_llm(backend, model, host):
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
        def call(prompt):
            r = client.messages.create(model=model, max_tokens=4000,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
    else:
        import requests
        def call(prompt):
            r = requests.post(f"{host}/api/chat", timeout=600, json={
                "model": model, "stream": False, "format": "json",
                "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"]
    return call


def sample_cell(rng):
    kind = rng.choices(list(TYPE_WEIGHTS), weights=list(TYPE_WEIGHTS.values()))[0]
    if kind == "ambiguous":
        rtype, clarity = "general_enquiry", "ambiguous"   # best-guess type for vague messages
        urgency, topic = rng.choice(AMBIGUOUS_URGENCY), "unclear (vague request)"
        quirk = rng.choice(QUIRKS)
    else:
        rtype, clarity = kind, "clear"
        urgency = rng.choice(ALLOWED_URGENCY[rtype])
        topic = rng.choice(SUB_TOPICS[rtype])
        quirk = rng.choice(QUIRKS)
        if rtype in ("complaint", "service_request", "general_enquiry") and rng.random() < .15:
            quirk = HARD_NEG_ROUTINE
        if rtype == "urgent_escalation" and rng.random() < .20:
            quirk = HARD_NEG_URGENT
    inv_hint = ", ".join(f"INV-{rng.randint(10000, 99999)}" for _ in range(6))
    return dict(rtype=rtype, urgency=urgency, clarity=clarity, topic=topic, inv_hint=inv_hint,
                tone=rng.choice(TONES), length=rng.choice(LENGTHS), quirk=quirk)


def parse_items(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return json.loads(text)["items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--calls", type=int, default=600)
    ap.add_argument("--per-call", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="change this when appending more data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/raw/raw.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cells = [sample_cell(rng) for _ in range(a.calls)]
    if a.dry_run:
        c = cells[0]
        print(PROMPT.format(n=a.per_call, rtype=c["rtype"], urgency=c["urgency"], topic=c["topic"], clarity=c["clarity"],
                            tone=c["tone"], length=c["length"], quirk=c["quirk"], inv_hint=c["inv_hint"]))
        from collections import Counter
        print("\nPlanned cells:", dict(Counter(f'{c["rtype"]}/{c["clarity"]}' for c in cells)))
        return

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    llm, lock, stats = make_llm(a.backend, a.model, a.host), threading.Lock(), {"ok": 0, "fail": 0}

    def work(idx_cell):
        idx, c = idx_cell
        prompt = PROMPT.format(n=a.per_call, rtype=c["rtype"], urgency=c["urgency"], topic=c["topic"], clarity=c["clarity"],
                               tone=c["tone"], length=c["length"], quirk=c["quirk"], inv_hint=c["inv_hint"])
        try:
            items = parse_items(llm(prompt))
        except Exception as e:
            with lock:
                stats["fail"] += 1
                print(f"[call {idx}] failed: {e}")
            return
        lo, hi = CONFIDENCE_RANGE[c["clarity"]]
        rows = []
        for j, it in enumerate(items):
            try:
                label = {"request_type": c["rtype"], "urgency": c["urgency"],
                         "sub_topic": it["sub_topic"], "confidence": round(random.uniform(lo, hi), 2),
                         "summary": it["summary"], "customer_sentiment": it["customer_sentiment"],
                         "entities": it["entities"], "required_action": it["required_action"]}
                rows.append({"id": f"s{a.seed}-{idx}-{j}", "source": "synthetic", "clarity": c["clarity"],
                             "cell": c, "subject": it["subject"], "message": it["message"], "label": label})
            except Exception:
                continue
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 20 == 0:
                print(f"{stats['ok']}/{a.calls} calls done ({stats['fail']} failed)")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, enumerate(cells)))
    print("Done:", stats, "->", a.out)


if __name__ == "__main__":
    main()