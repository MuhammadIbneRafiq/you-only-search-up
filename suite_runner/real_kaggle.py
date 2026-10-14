"""Real-data study (no synthetic parts): all 12 methods on Walmart, Favorita and M5 with their real descriptors,
the M2 quality measure and the same settings as the original real-data run (depth 3, coverage 2-15%, >= 2 stores,
top-20). Significance: a DFD of 500 runs on shuffled descriptors per method (Engelen eq. 4.6), the same 500 shuffles
for every method.

    from real_kaggle import run_real
    run_real("walmart")             # or "favorita" / "m5"

Results in <out_dir>/<dataset>/: methods.csv (one row per method), subgroups.csv (top-20 per method with coverage,
lift, p-value, ...), dfd_values.csv. Checkpointed per method; rerun (or attach the previous Kaggle output) to continue.
"""
import glob
import os
import shutil
import time
from pathlib import Path

import make_binary as mb
from binary_kaggle import REPO, _LEVELS, _cells, _find

REAL_METHODS = ["BS w=10", "BS w=50", "BS w=10 +local",
                "BUS", "BUS +local", "BUS-α", "BUS→BS pool",
                "BUS snn", "BUS snn +local", "BUS snn +local per level", "BUS→BS pool snn", "BUS→BS pool snn +local"]

REAL_CORE = r'''
def permuted_descriptors(rng):
    """Null model of the original real-data run: store-level descriptors shuffled between stores, item-level ones
    between items, record-level ones within each item."""
    perm = records[descriptor_cols].copy()
    for key, lvl in (("store", "store"), ("item", "item")):
        cols = [c for c in descriptor_cols if desc_level[c] == lvl]
        if cols:
            table = records.groupby(key)[cols].first()
            donor = dict(zip(table.index, rng.permutation(table.index)))
            perm[cols] = table.loc[records[key].map(donor)].to_numpy()
    for c in [c for c in descriptor_cols if desc_level[c] == "record"]:
        perm[c] = records.groupby("item")[c].transform(lambda s: rng.permutation(s.to_numpy()))
    return perm.astype(str).to_numpy(dtype=object)


def encode(A_obj):
    """Descriptor values -> integer codes per column (same code table for every shuffle: values are only moved)."""
    codes = np.empty(A_obj.shape, dtype=np.int16)
    for j in range(A_obj.shape[1]):
        codes[:, j] = np.searchsorted(VALUES[j], A_obj[:, j])
    return codes


def value_freqs_cat(A_):
    return [np.bincount(A_[:, j], minlength=len(VALUES[j])) / len(A_) for j in range(A_.shape[1])]


def enriched_cat(mc, A_, freqs, top, z_min=2.0):
    """Conditions shared by >= MAJORITY of the mini-cluster, ranked by binomial enrichment z (z >= z_min), at most top."""
    sub = A_[list(mc)]
    n = len(mc)
    conds = []
    for j in range(A_.shape[1]):
        values, counts = np.unique(sub[:, j], return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            s, base = c / n, freqs[j][v]
            if s >= MAJORITY and 0 < base < 1:
                z = (s - base) * np.sqrt(n) / np.sqrt(base * (1 - base))
                if z >= z_min:
                    conds.append((z, (j, v)))
    conds.sort(reverse=True)
    return [c for _, c in conds[:top]]


def run_real_method(name, A_, d, mcs, bs50_time=0.0):
    """One method on descriptor matrix A_ (integer codes); mini-clusters depend on the target only, so they are shared
    by the real run and every shuffle. Returns (top, info)."""
    t0 = time.time()
    phi = MonetaryQuality(excess, level, store_codes, relative=True)
    freqs = value_freqs_cat(A_)
    info = {}
    base = name.replace(" +local per level", "").replace(" +local", "")

    def bart(pairs):
        budget = max(BART_MIN_TIME, bs50_time)
        top_, st, finished, _ = bart_exact(pairs, A_, d, phi, budget)
        info.update(finished=finished, time_budget_s=budget)
        return top_

    def snn_baskets():
        return [enriched_cat(mc, A_, freqs, max(TOP_ENRICHED, d)) for mc in mcs["snn"]]

    def pool(baskets):
        votes = Counter(c for b in baskets for c in b)
        return beam_search(A_, phi, w=POOL_W, d=d, prune=True, pool=[c for c, _ in votes.most_common(POOL_SIZE)])[0]

    if name.endswith("+local per level"):
        top, _ = bus_levels_local(A_, phi, snn_baskets(), d)
    elif base.startswith("BS"):
        top = beam_search(A_, phi, w=int(base.split("w=")[1]), d=d, prune=True)[0]
    elif base == "BUS":
        top = bart(mcs["closest"])
    elif base == "BUS-α":
        top = bart(mcs["alpha"])
    elif base == "BUS→BS pool":
        top = pool([[(int(j), int(v)) for j, v in shared_conditions(list(p), A_)] for p in mcs["alpha"]])
    elif base == "BUS snn":
        cands = set()
        for b in snn_baskets():
            for size in range(1, min(d, len(b)) + 1):
                cands.update(tuple(sorted(c)) for c in itertools.combinations(b, size))
        top = evaluate_candidates(A_, phi, cands, prune=True)[0]
    elif base == "BUS→BS pool snn":
        top = pool(snn_baskets())
    else:
        raise ValueError(name)
    if name.endswith(" +local"):
        rs = ResultSet(Q)
        for q_, d_, m_ in top:
            rs.insert(q_, d_, m_)
        top = local_search_d(A_, phi, top, rs, Counter(), d)
    info.update(runtime_s=time.time() - t0, evaluated=len(phi.cache))
    return top, info


def _dfd_chunk(name, A_list, d, mcs, bs50_time):
    return [(run_real_method(name, A_p, d, mcs, bs50_time)[0] or [(0.0,)])[0][0] for A_p in A_list]


def describe_codes(desc):
    return " AND ".join(f"{descriptor_cols[j]} = {VALUES[j][v]}" for j, v in desc)


def subgroup_rows(name, top, dfd):
    """Top-q subgroups with the metrics of the original real-data run (Konijn money metrics, Engelen alpha / beta /
    phi_cluster and lift on the scale-free target, coverage, complement) and the DFD p-value."""
    m2 = MonetaryQuality(excess, level, store_codes, relative=True)
    rows = []
    for rank, (quality, desc, mask) in enumerate(top, start=1):
        idx = np.flatnonzero(mask)
        n, n_c = len(idx), N - len(idx)
        within = D_shape[np.ix_(idx, idx)].sum(dtype=np.float64)
        alpha = (SHAPE_ROWSUM[idx].sum() - within) / (n * n_c)
        beta = within / (n * (n - 1))
        X_s, L_s = X_TOT[mask].sum(), level[mask].sum()
        p = ((dfd > quality).sum() + 0.5 * (dfd == quality).sum() + 1) / (len(dfd) + 1)
        rows.append({"dataset": DATASET, "method": name, "rank": rank, "description": describe_codes(desc),
                     "length": len(desc), "quality": quality, "excess_raw": X_s, "excess_relative": m2.signed(mask),
                     "pct_of_normal": 100 * X_s / (L_s * T),
                     "complement_pct_of_normal": 100 * (X_TOT.sum() - X_s) / ((level.sum() - L_s) * T),
                     "coverage_n": n, "coverage": n / N, "n_stores": m2.n_stores(mask), "complement_n": n_c,
                     "alpha": alpha, "beta": beta, "phi_cluster": alpha / (beta + 1), "lift": SHAPE_OVERALL / beta,
                     "p_value": p})
    return rows
'''

REAL_RUN = r'''
# ---- setup shared by every method: integer codes, mini-clusters (target only), the 500 shuffles
VALUES = [np.array(sorted(set(A[:, j]))) for j in range(A.shape[1])]
A_codes = encode(A)
d = D_MAX
max_size = int(MAX_COV * N)
D_shape = pairwise_cityblock(rel_dev)                                   # for alpha / beta / lift in the subgroup table
SHAPE_ROWSUM = D_shape.sum(axis=1, dtype=np.float64)
SHAPE_OVERALL = SHAPE_ROWSUM.sum() / (N * (N - 1))
X_TOT = excess.sum(axis=1, dtype=np.float64)
D_bart = pairwise_sqeuclidean(excess)
MCS = {"closest": top_pairs(D_bart, U_SEEDS, "closest"), "alpha": top_pairs(D_bart, U_SEEDS, "alpha"),
       "snn": snn_components_exp(pairwise_cityblock(block_means_n(rel_dev)), max_size)}
del D_bart
print(f"{DATASET}: {N} series, {A.shape[1]} descriptors, {sum(len(v) for v in VALUES)} conditions, depth {d}; "
      f"mini-clusters: {len(MCS['closest'])} pairs, {len(MCS['snn'])} SNN groups")
rng = np.random.default_rng(SEED)
permuted = [encode(permuted_descriptors(rng)) for _ in range(NU)]      # the same shuffles for every method
chunks = [permuted[i::N_JOBS] for i in range(N_JOBS)]

methods_csv, subgroups_csv = f"{RESULTS_DIR}/methods.csv", f"{RESULTS_DIR}/subgroups.csv"
done = set(pd.read_csv(methods_csv)["method"]) if os.path.exists(methods_csv) else set()
print(f"{len(done)} of {len(METHODS_R)} methods already done")
bs50_time = float(pd.read_csv(methods_csv).set_index("method").runtime_s.get("BS w=50", 0.0)) if done else 0.0
for name in METHODS_R:
    if name in done:
        continue
    if time.time() > DEADLINE:
        print("time budget reached; rerun to continue")
        break
    top, info = run_real_method(name, A_codes, d, MCS, bs50_time)
    if name == "BS w=50":
        bs50_time = info["runtime_s"]
    t = time.time()
    parts = Parallel(n_jobs=N_JOBS)(delayed(_dfd_chunk)(name, chunk, d, MCS, bs50_time) for chunk in chunks)
    dfd = np.array([v for part in parts for v in part])
    rows = subgroup_rows(name, top, dfd)
    p = np.array([r["p_value"] for r in rows]) if rows else np.array([1.0])
    row = {"dataset": DATASET, "method": name, "nu": NU, "best_quality": top[0][0] if top else np.nan,
           "mean_quality_top20": float(np.mean([q for q, _, _ in top])) if top else np.nan,
           "n_significant_005": int((p < 0.05).sum()), "n_significant_001": int((p < 0.01).sum()),
           "dfd_median": float(np.median(dfd)), "dfd_95": float(np.quantile(dfd, 0.95)),
           "dfd_99": float(np.quantile(dfd, 0.99)), "dfd_seconds": time.time() - t, **info}
    pd.DataFrame(rows).to_csv(subgroups_csv, mode="a", header=not os.path.exists(subgroups_csv), index=False)
    pd.DataFrame({name: dfd}).to_csv(f"{RESULTS_DIR}/dfd_{METHODS_R.index(name):02d}.csv", index=False)
    new = pd.DataFrame([row])
    if os.path.exists(methods_csv):
        new = pd.concat([pd.read_csv(methods_csv), new], ignore_index=True)
    new.to_csv(methods_csv, index=False)
    print(f"{name}: best {row['best_quality']:.4g}, {row['n_significant_005']} of {len(rows)} with p < 0.05, "
          f"search {info['runtime_s']:.1f}s, DFD {row['dfd_seconds']:.0f}s")
'''

REAL_SUMMARY = r'''
methods = pd.read_csv(f"{RESULTS_DIR}/methods.csv").set_index("method").reindex([m for m in METHODS_R])
display(methods[["best_quality", "mean_quality_top20", "n_significant_005", "n_significant_001", "dfd_95", "dfd_99",
                 "evaluated", "runtime_s", "dfd_seconds"]].round(4))
sub = pd.read_csv(f"{RESULTS_DIR}/subgroups.csv")
display(sub[sub["rank"] <= 5][["method", "rank", "description", "quality", "coverage", "n_stores", "pct_of_normal",
                               "lift", "p_value"]].round(4))
'''


def _copy_previous(out_dir, dataset, results_name, base):
    """Carry over finished methods from an attached earlier Kaggle output (or an earlier session in out_dir)."""
    target = Path(out_dir) / dataset
    target.mkdir(parents=True, exist_ok=True)
    if (target / "methods.csv").exists():
        return
    hits = glob.glob(f"{base}/**/{results_name}/{dataset}/methods.csv", recursive=True)
    if hits:
        src = Path(hits[0]).parent
        for f in src.iterdir():
            shutil.copy(f, target / f.name)
        print("resume: copied finished methods from", src)


def run_real(dataset, hours=11.0, methods=None, data_base="/kaggle/input", out_dir="/kaggle/working/results_real",
             n_jobs=None, nu=None):
    """Run (or continue) the real-data study for one dataset; returns the namespace."""
    deadline = time.time() + hours * 3600
    out_dir = str(Path(out_dir).resolve())
    data_root = _find("final_sales_dataset.csv", data_base)
    assert data_root is not None, f"final_sales_dataset.csv not found under {data_base}"
    data_root = data_root.parent
    _copy_previous(out_dir, dataset, Path(out_dir).name, data_base)

    template = _cells(REPO / "m5_deep_v5.ipynb")
    if dataset != "m5":
        loader = _cells(REPO / f"{dataset}_subtle.ipynb")
        template[2], template[6] = loader[2], loader[3]
    scratch = Path(out_dir) / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    from IPython.display import display
    ns = {"__name__": "real_study", "display": display}
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        for i in range(1, 14):
            if i == 6:
                os.chdir(data_root)
            exec(compile(template[i], f"<template cell {i}>", "exec"), ns)
            if i == 6:
                os.chdir(scratch)
            if i == 2:
                ns["M5_DIR"] = str(data_root / "M5")
        ns["QUICK"] = False
        exec(mb.SETTINGS.format(methods=[]), ns)                 # the shared constants (pool width, Bart's budget, ...)
        exec(mb.CORE + _LEVELS + REAL_CORE, ns)
        ns.update(RESULTS_DIR=f"{out_dir}/{dataset}", METHODS_R=list(methods or REAL_METHODS), DEADLINE=deadline)
        os.makedirs(ns["RESULTS_DIR"], exist_ok=True)
        if n_jobs:
            ns["N_JOBS"] = n_jobs
        if nu:
            ns["NU"] = nu
        exec(REAL_RUN, ns)
        exec(REAL_SUMMARY, ns)
    finally:
        os.chdir(cwd)
    shutil.rmtree(scratch, ignore_errors=True)
    return ns
