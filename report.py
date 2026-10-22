"""Build comparison table + charts from results/*.json.

  python report.py --names baseline finetuned
  python report.py --names baseline finetuned --ablation-prefix ft_n   # also plots results/ft_n<N>.json vs N
"""
import argparse, glob, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import REQUEST_TYPES

ROWS = [("workflow_valid_rate", "Accepted by workflow's own parser", 1), ("strict_schema_rate", "Full schema compliance", 1), ("type_accuracy", "Type accuracy (clear)", 1),
        ("type_macro_f1", "Type macro-F1 (clear)", 1), ("urgency_accuracy", "Urgency accuracy (clear)", 1),
        ("urgency_macro_f1", "Urgency macro-F1 (clear)", 1), ("emergency_recall", "Emergency recall", 1),
        ("false_emergency_rate", "False-emergency rate (lower better)", 1),
        ("handwritten_type_accuracy", "HAND-WRITTEN subset: type accuracy", 1),
        ("handwritten_emergency_recall", "HAND-WRITTEN subset: emergency recall", 1),
        ("handwritten_ambiguity_recall", "HAND-WRITTEN subset: ambiguity recall", 1),
        ("entity_invoice_number_match", "Entity: invoice match", 1), ("entity_amount_match", "Entity: amount match", 1),
        ("entity_currency_match", "Entity: currency match", 1), ("hallucinated_entity_rate", "Hallucinated entities (lower better)", 1),
        ("ambiguity_recall", "Ambiguity recall (sent to review)", 1), ("ambiguity_precision", "Ambiguity precision", 1),
        ("false_review_rate_on_clear", "Needless review on clear items (lower better)", 1),
        ("human_review_rate", "Overall human-review rate", 1), ("ece_type_on_clear", "ECE (secondary, lower better)", 0),
        ("latency_mean_s", "Latency mean (s)", 0), ("latency_p95_s", "Latency p95 (s)", 0),
        ("prompt_tokens_mean", "Prompt tokens (mean)", 0)]


def fmt(v, pct):
    if v is None: return "n/a"
    return f"{v*100:.1f}%" if pct else f"{v:.3g}"


def confusion(ax, items, name):
    idx = {c: i for i, c in enumerate(REQUEST_TYPES + ["invalid"])}
    M = np.zeros((4, 5), int)
    for it in items:
        if it["clarity"] != "clear": continue
        M[idx[it["gold_type"]], idx[it["pred_type"]] if it["pred_type"] else 4] += 1
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(REQUEST_TYPES + ["invalid"], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(REQUEST_TYPES, fontsize=8)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, M[i, j], ha="center", va="center", color="black" if M[i, j] < M.max() * .6 else "white")
    ax.set_title(f"{name}: type confusion (clear items)", fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("gold")


def reliability(ax, items, name, bins=10):
    vi = [i for i in items if i["clarity"] == "clear" and i["valid"]]
    xs, ys, ns = [], [], []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        s = [i for i in vi if lo <= i["confidence"] < hi or (b == bins - 1 and i["confidence"] == 1.0)]
        if s:
            xs.append(np.mean([i["confidence"] for i in s])); ys.append(np.mean([i["type_correct"] for i in s])); ns.append(len(s))
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.scatter(xs, ys, s=[20 + n for n in ns]); ax.plot(xs, ys)
    ax.set_title(f"{name}: reliability (clear items)", fontsize=9); ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)


def conf_hist(ax, items, name):
    for cl, col in (("clear", "tab:blue"), ("ambiguous", "tab:orange")):
        v = [i["confidence"] for i in items if i["clarity"] == cl and i["valid"]]
        ax.hist(v, bins=np.linspace(0, 1, 21), alpha=.6, label=cl, color=col)
    ax.axvline(0.70, color="red", ls="--", label="review threshold")
    ax.set_title(f"{name}: confidence, clear vs ambiguous", fontsize=9); ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--ablation-prefix", default=None)
    a = ap.parse_args()
    res = {n: json.load(open(f"results/{n}.json", encoding="utf-8")) for n in a.names}

    lines = ["| Metric | " + " | ".join(a.names) + " |", "|---|" + "---|" * len(a.names)]
    for k, label, pct in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(res[n]["metrics"].get(k), pct) for n in a.names) + " |")
    first = res[a.names[0]]["metrics"]
    lines.append(f"\nTest set: {first['n_test']} items ({first['n_clear']} clear, {first['n_ambiguous']} ambiguous); "
                 f"entity metrics over {first['n_entity_items']} items.")
    open("results/comparison.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))

    for fn, tag in ((confusion, "confusion"), (reliability, "reliability"), (conf_hist, "confidence_hist")):
        fig, axes = plt.subplots(1, len(a.names), figsize=(5.2 * len(a.names), 4.2))
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, a.names):
            fn(ax, res[n]["items"], n)
        fig.tight_layout(); fig.savefig(f"results/{tag}.png", dpi=150); plt.close(fig)

    if a.ablation_prefix:
        pts = []
        for p in glob.glob(f"results/{a.ablation_prefix}*.json"):
            mm = re.search(rf"{a.ablation_prefix}(\d+)\.json$", p)
            if mm: pts.append((int(mm.group(1)), json.load(open(p, encoding="utf-8"))["metrics"]))
        pts.sort()
        if pts:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for key, lab in (("type_macro_f1", "type macro-F1"), ("urgency_macro_f1", "urgency macro-F1"), ("emergency_recall", "emergency recall")):
                ax.plot([p[0] for p in pts], [p[1][key] for p in pts], marker="o", label=lab)
            if "baseline" in res:
                ax.axhline(res["baseline"]["metrics"]["type_macro_f1"], ls="--", color="gray", label="baseline type F1")
            ax.set_xscale("log"); ax.set_xlabel("training examples"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
            ax.set_title("Effect of training-set size"); fig.tight_layout(); fig.savefig("results/ablation.png", dpi=150)
    print("\ncharts saved in results/")


if __name__ == "__main__":
    main()