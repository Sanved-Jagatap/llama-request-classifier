"""Score a predictions file against data/test.jsonl.

  python evaluate.py --preds preds/baseline.jsonl --name baseline
Writes results/<name>.json (metrics + per-item details).
Type/urgency metrics are computed on CLEAR items; ambiguous items are scored by
ambiguity detection (did the system route them to human review?).
"""
import argparse, json, re, statistics
from collections import Counter
from common import REQUEST_TYPES, URGENCIES, SENTIMENTS, LABEL_KEYS

REVIEW_THRESHOLD = 0.70


def parse_output(raw):
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    for cand in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0] if re.search(r"\{.*\}", t, re.S) else None):
        if cand:
            try:
                o = json.loads(cand)
                return o if isinstance(o, dict) else None
            except Exception:
                pass
    return None


def schema_ok(o):
    if o is None or any(k not in o for k in LABEL_KEYS):
        return False
    if o["request_type"] not in REQUEST_TYPES or o["urgency"] not in URGENCIES:
        return False
    if o["customer_sentiment"] not in SENTIMENTS:
        return False
    if isinstance(o["confidence"], bool) or not isinstance(o["confidence"], (int, float)) or not 0 <= o["confidence"] <= 1:
        return False
    e = o["entities"]
    return isinstance(e, dict) and all(k in e for k in ("invoice_number", "amount", "currency"))


def wf_ok(o):
    """Same acceptance rule as the n8n 'Parse & Validate Classification' node."""
    return (isinstance(o, dict) and o.get("request_type") in REQUEST_TYPES and o.get("urgency") in URGENCIES
            and isinstance(o.get("confidence"), (int, float)) and not isinstance(o.get("confidence"), bool)
            and 0 <= o["confidence"] <= 1)


def invalid_reason(o):
    if o is None: return "not parseable JSON"
    if not isinstance(o, dict): return "not an object"
    if o.get("request_type") not in REQUEST_TYPES: return "bad/missing request_type"
    if o.get("urgency") not in URGENCIES: return "bad/missing urgency"
    c = o.get("confidence")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1: return "bad/missing confidence"
    miss = [k for k in LABEL_KEYS if k not in o]
    if miss: return "missing keys (strict only)"
    if o["customer_sentiment"] not in SENTIMENTS: return "bad sentiment (strict only)"
    e = o["entities"]
    if not isinstance(e, dict) or not all(k in e for k in ("invoice_number", "amount", "currency")): return "bad entities shape (strict only)"
    return None


def norm_amount(x):
    if x is None:
        return None
    try:
        return f"{float(str(x).replace(',', '')):g}"
    except Exception:
        return str(x).strip()


def norm_ent(field, x):
    if x is None or str(x).strip() == "":
        return None
    return norm_amount(x) if field == "amount" else str(x).strip().upper()


def macro_f1(gold, pred, classes):
    f1s = {}
    for c in classes:
        if not any(g == c or p == c for g, p in zip(gold, pred)):
            continue  # class absent from both gold and predictions
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return (sum(f1s.values()) / len(f1s) if f1s else 0.0), f1s


def ece(confs, correct, bins=10):
    if not confs:
        return None
    tot, n = 0.0, len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if idx:
            tot += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    a = ap.parse_args()

    test = {r["id"]: r for r in (json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip())}
    preds = {p["id"]: p for p in (json.loads(l) for l in open(a.preds, encoding="utf-8") if l.strip())}
    items = []
    for rid, rec in test.items():
        p = preds.get(rid, {"raw": ""})
        o = parse_output(p.get("raw"))
        valid = wf_ok(o)            # what the real workflow would accept
        strict = schema_ok(o)       # full schema compliance
        g = rec["label"]
        conf = float(o["confidence"]) if valid else None
        review = (not valid) or conf < REVIEW_THRESHOLD
        pt, pu = (o["request_type"], o["urgency"]) if valid else (None, None)
        items.append({"id": rid, "source": rec["source"], "clarity": rec["clarity"],
                      "gold_type": g["request_type"], "pred_type": pt,
                      "gold_urgency": g["urgency"], "pred_urgency": pu,
                      "confidence": conf, "valid": valid, "strict_valid": strict, "reason": invalid_reason(o), "review": review,
                      "type_correct": pt == g["request_type"],
                      "latency_s": p.get("latency_s"), "prompt_tokens": p.get("prompt_tokens"),
                      "gold_entities": g.get("entities"), "pred_entities": (o.get("entities") if isinstance(o.get("entities"), dict) else {}) if valid else None,
                      "message": rec["message"]})

    n = len(items)
    clear = [i for i in items if i["clarity"] == "clear"]
    amb = [i for i in items if i["clarity"] == "ambiguous"]
    m = {"name": a.name, "n_test": n, "n_clear": len(clear), "n_ambiguous": len(amb)}
    m["workflow_valid_rate"] = sum(i["valid"] for i in items) / n
    m["strict_schema_rate"] = sum(i["strict_valid"] for i in items) / n
    m["invalid_reasons"] = dict(Counter(i["reason"] for i in items if i["reason"]))

    gt = [i["gold_type"] for i in clear]; pt = [i["pred_type"] for i in clear]
    m["type_accuracy"] = sum(g == p for g, p in zip(gt, pt)) / max(1, len(clear))
    m["type_macro_f1"], m["type_f1_per_class"] = macro_f1(gt, pt, REQUEST_TYPES)
    gu = [i["gold_urgency"] for i in clear]; pu = [i["pred_urgency"] for i in clear]
    m["urgency_accuracy"] = sum(g == p for g, p in zip(gu, pu)) / max(1, len(clear))
    m["urgency_macro_f1"], m["urgency_f1_per_class"] = macro_f1(gu, pu, URGENCIES)

    # emergency handling: the workflow forces the urgent branch on type OR critical
    emerg_gold = [i for i in clear if i["gold_type"] == "urgent_escalation"]
    caught = lambda i: i["valid"] and (i["pred_type"] == "urgent_escalation" or i["pred_urgency"] == "critical")
    m["emergency_recall"] = sum(caught(i) for i in emerg_gold) / max(1, len(emerg_gold))
    non = [i for i in clear if i["gold_type"] != "urgent_escalation"]
    m["false_emergency_rate"] = sum(caught(i) for i in non) / max(1, len(non))

    # honest subset: hand-written items only (not produced by the data-generating LLM)
    hc = [i for i in clear if i["source"] == "handwritten"]
    ha = [i for i in amb if i["source"] == "handwritten"]
    m["n_handwritten_clear"], m["n_handwritten_ambiguous"] = len(hc), len(ha)
    m["handwritten_type_accuracy"] = sum(i["type_correct"] for i in hc) / len(hc) if hc else None
    he = [i for i in hc if i["gold_type"] == "urgent_escalation"]
    m["handwritten_emergency_recall"] = sum(caught(i) for i in he) / len(he) if he else None
    m["handwritten_ambiguity_recall"] = sum(i["review"] for i in ha) / len(ha) if ha else None

    # entities (only items that have gold entities)
    ge = [i for i in items if i["gold_entities"] is not None]
    for f in ("invoice_number", "amount", "currency"):
        ok = [norm_ent(f, i["gold_entities"].get(f)) == norm_ent(f, (i["pred_entities"] or {}).get(f)) for i in ge if i["valid"] or True]
        m[f"entity_{f}_match"] = sum(ok) / max(1, len(ok))
    m["n_entity_items"] = len(ge)
    def halluc(i):
        if not i["valid"]:
            return False
        e, t = i["pred_entities"], i["message"].replace(",", "")
        bad = (e.get("invoice_number") not in (None, "") and str(e["invoice_number"]) not in i["message"]) or \
              (e.get("amount") not in (None, "") and norm_amount(e["amount"]) not in {norm_amount(x) for x in re.findall(r"\d+(?:\.\d+)?", t)})
        return bad
    m["hallucinated_entity_rate"] = sum(halluc(i) for i in items) / n

    # human-review behaviour
    m["human_review_rate"] = sum(i["review"] for i in items) / n
    m["false_review_rate_on_clear"] = sum(i["review"] for i in clear) / max(1, len(clear))
    tp = sum(i["review"] for i in amb)
    m["ambiguity_recall"] = tp / max(1, len(amb))
    m["ambiguity_precision"] = tp / max(1, sum(i["review"] for i in items))

    # calibration (secondary)
    vi = [i for i in clear if i["valid"]]
    m["ece_type_on_clear"] = ece([i["confidence"] for i in vi], [int(i["type_correct"]) for i in vi])
    cc = [i["confidence"] for i in clear if i["valid"]]; ca = [i["confidence"] for i in amb if i["valid"]]
    m["mean_conf_clear"] = statistics.mean(cc) if cc else None
    m["mean_conf_ambiguous"] = statistics.mean(ca) if ca else None

    lat = sorted(i["latency_s"] for i in items if i["latency_s"] is not None)
    if lat:
        m["latency_mean_s"] = statistics.mean(lat)
        m["latency_p95_s"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    pts = [i["prompt_tokens"] for i in items if i["prompt_tokens"]]
    m["prompt_tokens_mean"] = statistics.mean(pts) if pts else None

    import os
    os.makedirs("results", exist_ok=True)
    json.dump({"metrics": m, "items": items}, open(f"results/{a.name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if not k.endswith("per_class")}, indent=1))


if __name__ == "__main__":
    main()