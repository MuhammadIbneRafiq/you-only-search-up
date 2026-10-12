# ---- Results vs beam search w=10 on the same 480 tasks (exact McNemar test), from the saved results (no rerun)
import pandas as pd
from scipy.stats import binomtest

deep = pd.read_csv("results_v5/m5_deep.csv")
Q, K_ATTRS, LITS_V5 = 20, [100, 400], [3, 4, 5, 6]
METHODS_V5 = ["BS w=10", "BS w=50", "BUS snn-comp", "BUS snn-comp +local", "BUS→BS pool snn-comp"]

deep["found"] = deep["rank"] <= Q
REF = "BS w=10"
key = ["k", "lits", "snr", "outliers", "rep"]
w = deep.pivot_table(index=key, columns="method", values="found").astype(bool)

# alpha is split over the tests in a family (Bonferroni), so running many small tests does not create false wins
families = [([("all", w)], 0.05),
            ([(f"lits={l}", w.xs(l, level="lits", drop_level=False)) for l in LITS_V5], 0.05 / len(LITS_V5)),
            ([(f"k={k} ({2 * k} conditions)", w.xs(k, level="k", drop_level=False)) for k in K_ATTRS], 0.05 / len(K_ATTRS))]
rows = []
for groups, alpha in families:
    for g, sub in groups:
        for m in [x for x in METHODS_V5 if x != REF]:
            wins, losses = int((sub[m] & ~sub[REF]).sum()), int((~sub[m] & sub[REF]).sum())
            p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
            if p < alpha:
                verdict = "better than BS w=10" if wins > losses else "worse than BS w=10"
            elif p < 0.05:
                verdict = "suggestive (not after correction)"
            else:
                verdict = "no significant difference"
            rows.append({"group": g, "method": m, "tasks": len(sub), "found": sub[m].mean(), "BS w=10": sub[REF].mean(),
                         "wins": wins, "losses": losses, "p": p, "alpha": alpha, "verdict": verdict})
results_v5 = pd.DataFrame(rows)

print("Paired comparison with beam search w=10 (wins = only this method found it, losses = only BS w=10 found it)")
print(results_v5.round(3))
print("Average per task")
print(deep.groupby("method")[["found", "evaluated", "runtime_s"]].mean().reindex(METHODS_V5).round(3))
