"""Validate, de-duplicate, split and format the dataset.

  python build_dataset.py
Outputs: data/train.jsonl, data/val.jsonl (chat format for fine-tuning)
         data/test.jsonl (raw records with gold labels, for evaluation)
Hand-written seed examples go to the TEST set only.
"""
import argparse, json, os, random, re
from collections import Counter, defaultdict
from common import *


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm_tokens(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def dedupe(recs, thr=0.8):
    kept, toks = [], []
    for r in recs:
        t = norm_tokens(r["message"])
        if any(len(t & o) / max(1, len(t | o)) >= thr for o in toks):
            continue
        kept.append(r); toks.append(t)
    return kept


def load_verdicts(path="data/raw/verdicts.jsonl"):
    import hashlib
    if not os.path.exists(path):
        return {}
    return {d["key"]: d["verifier_type"] for d in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/raw.jsonl")
    ap.add_argument("--handwritten", default="data/seed/handwritten.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    raw = read_jsonl(a.raw)
    good, bad, fixed = [], Counter(), Counter()
    for r in raw:
        r, fixes = sanitize_record(r)
        for fx in set(fixes): fixed[fx] += 1
        errs = validate_record(r)
        if errs:
            for e in errs: bad[e] += 1
        else:
            good.append(r)
    print(f"raw={len(raw)}  valid={len(good)}  rejected={len(raw)-len(good)}")
    for fx, n in fixed.most_common(): print(f"   auto-fixed: {fx}: {n}")
    for e, n in bad.most_common(): print(f"   rejected: {e}: {n}")

    # hand-written text must never leak into train/val
    hand = read_jsonl(a.handwritten)
    hand_toks = [norm_tokens(h["message"]) for h in hand]
    good = [r for r in good if not any(len(norm_tokens(r["message"]) & t) / max(1, len(norm_tokens(r["message"]) | t)) >= .8 for t in hand_toks)]
    verdicts = load_verdicts()
    if verdicts:
        import hashlib
        kept, disagree, unverified = [], Counter(), 0
        for r in good:
            if r["clarity"] != "clear":
                kept.append(r); continue
            v = verdicts.get(hashlib.sha1(r["message"].strip().encode("utf-8")).hexdigest())
            if v is None:
                unverified += 1; kept.append(r)
            elif v == r["label"]["request_type"]:
                kept.append(r)
            else:
                disagree[f'{r["label"]["request_type"]} -> judge says {v}'] += 1
        n_clear = sum(r["clarity"] == "clear" for r in good)
        print(f"verifier: dropped {sum(disagree.values())} of {n_clear} clear rows ({unverified} unverified kept)")
        for k, n in disagree.most_common(): print(f"   {k}: {n}")
        good = kept
    good = dedupe(good)
    # small models love reusing the same invoice numbers: keep at most 2 rows per invoice number
    seen_inv, kept = Counter(), []
    for r in good:
        inv = r["label"]["entities"]["invoice_number"]
        if inv:
            seen_inv[inv] += 1
            if seen_inv[inv] > 2:
                continue
        kept.append(r)
    print(f"dropped for repeated invoice numbers: {len(good) - len(kept)}")
    good = kept
    print(f"after dedupe + leakage filter: {len(good)}")

    strata = defaultdict(list)
    for r in good:
        strata[f'{r["label"]["request_type"]}|{r["clarity"]}'].append(r)
    train, val, test = [], [], []
    for k, rows in strata.items():
        rng.shuffle(rows)
        n = len(rows); nv = max(1, round(n * .1)) if n >= 10 else 0
        train += rows[nv * 2:]; val += rows[:nv]; test += rows[nv:nv * 2]

    for h in hand:  # hand-written -> test only
        test.append({"id": f"h-{len(test)}", "source": "handwritten",
                     "clarity": "ambiguous" if h.get("ambiguous") else "clear",
                     "subject": h["subject"], "message": h["message"],
                     "label": {"request_type": h["request_type"], "urgency": h["urgency"],
                               "customer_sentiment": h.get("customer_sentiment"),
                               "entities": h.get("entities")}})
    rng.shuffle(train); rng.shuffle(val)
    os.makedirs(a.out, exist_ok=True)
    write_jsonl(f"{a.out}/train.jsonl", [to_chat_example(r) for r in train])
    write_jsonl(f"{a.out}/val.jsonl", [to_chat_example(r) for r in val])
    write_jsonl(f"{a.out}/test.jsonl", test)

    print("\nsplit sizes:", {"train": len(train), "val": len(val), "test": len(test),
                             "test_handwritten": len(hand)})
    for name, rows in (("train", train), ("val", val), ("test", test)):
        c = Counter(r["label"]["request_type"] for r in rows)
        print(f"{name:5s}", dict(c), "| ambiguous:", sum(r["clarity"] == "ambiguous" for r in rows))


if __name__ == "__main__":
    main()