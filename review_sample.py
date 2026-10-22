"""Manual quality review of synthetic data.
  python review_sample.py --make     # writes data/review_sample.csv (random sample of train)
  (open the CSV, fill the 'ok' column with y or n, and optionally a note)
  python review_sample.py --tally    # prints label-error rate for your data card
"""
import argparse, csv, json, random
ap = argparse.ArgumentParser(); ap.add_argument("--make", action="store_true"); ap.add_argument("--tally", action="store_true"); ap.add_argument("--n", type=int, default=80, help="rows to review")
a = ap.parse_args(); path = "data/review_sample.csv"
if a.make:
    rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    random.Random(7).shuffle(rows); rows = rows[:a.n]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["message", "request_type", "urgency", "sentiment", "sub_topic", "ok(y/n)", "note"])
        for r in rows:
            u = r["messages"][1]["content"]; l = json.loads(r["messages"][2]["content"])
            w.writerow([u, l["request_type"], l["urgency"], l["customer_sentiment"], l["sub_topic"], "", ""])
    print(f"wrote {len(rows)} rows to {path}")
if a.tally:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    done = [r for r in rows if r["ok(y/n)"].strip().lower() in ("y", "n")]
    bad = [r for r in done if r["ok(y/n)"].strip().lower() == "n"]
    print(f"reviewed {len(done)}/{len(rows)}; label errors: {len(bad)} ({100*len(bad)/max(1,len(done)):.1f}%)")