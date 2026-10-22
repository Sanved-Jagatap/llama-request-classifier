"""Did the label judge unfairly remove the hard cases? Run after verify_labels.py:  python diagnose.py"""
import argparse, hashlib, json, random
from collections import defaultdict
from common import HARD_NEG_ROUTINE, HARD_NEG_URGENT

ap = argparse.ArgumentParser()
ap.add_argument("--show", default=None, help="print dropped rows for this design type, e.g. general_enquiry")
ap.add_argument("--n", type=int, default=8)
args = ap.parse_args()
key = lambda m: hashlib.sha1(m.strip().encode("utf-8")).hexdigest()
verd = {d["key"]: d["verifier_type"] for d in map(json.loads, open("data/raw/verdicts.jsonl", encoding="utf-8"))}
rows = [json.loads(l) for l in open("data/raw/raw.jsonl", encoding="utf-8") if l.strip()]
stat = defaultdict(lambda: [0, 0])   # group -> [judged, agreed]
for r in rows:
    v = verd.get(key(r["message"]))
    if r.get("clarity") != "clear" or v is None:
        continue
    t, q = r["label"]["request_type"], r["cell"]["quirk"]
    groups = [f"type: {t}"]
    if q == HARD_NEG_ROUTINE: groups.append("HARD: routine issue, urgent-sounding words")
    if q == HARD_NEG_URGENT:  groups.append("HARD: calm tone, genuine emergency")
    for g in groups:
        stat[g][0] += 1; stat[g][1] += int(v == t)
print(f"{'group':48s} judged  agree%")
for g, (n, a) in sorted(stat.items()):
    print(f"{g:48s} {n:6d}  {100*a/n:5.1f}%")

if args.show:
    bad = [(r, verd[key(r["message"])]) for r in rows
           if r.get("clarity") == "clear" and r["label"]["request_type"] == args.show
           and key(r["message"]) in verd and verd[key(r["message"])] != args.show]
    random.Random(1).shuffle(bad)
    print(f"\n{len(bad)} rows designed as {args.show} were disagreed with. Random {args.n}:\n")
    for r, v in bad[:args.n]:
        print(f"[designed {args.show} | judge says {v} | quirk: {r['cell']['quirk']}]")
        print("  " + r["message"].replace("\n", " ")[:400] + "\n")