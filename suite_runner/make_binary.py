"""Binary semi-synthetic study: builds binary_<dataset>.ipynb for M5, Walmart and Favorita from m5_deep_v5.ipynb.
Identical algorithms, quality measure (M2), constraints and task design on every dataset; only the target differs.
Usage: python make_binary.py <quick|full> [dataset ...]"""
import sys

DATASETS = ["m5", "walmart", "favorita"]

METHODS = ["BS w=10", "BS w=50",
           "BUS", "BUS +local", "BUS-α", "BUS→BS pool",                       # Bart's seeds (closest / alpha pairs)
           "BUS snn", "BUS snn +local", "BUS→BS pool snn"]                    # fixed seeds (SNN components)

SETTINGS = '''
# ================= binary study: settings =================
K_ATTRS = [20, 400] if QUICK else [20, 50, 100, 400]   # binary attributes (2 conditions each)
LITS_B = [2, 10] if QUICK else [2, 4, 6, 8, 10]         # conditions in the planted (true) description
GRID = [(8, 0.0)] if QUICK else [(8, 0.0), (8, 0.2), (16, 0.0), (16, 0.2)]   # (snr, outliers)
REPS_B = 1 if QUICK else 10
P_RANGE = (0.5, 0.9)              # share of records with value 1, drawn per attribute (uneven binary attributes)
COVERAGE_RANGE = (0.01, 0.15)     # accepted coverage of a planted subgroup (= the min / max coverage constraints)
MIN_COV_B = 0.01                  # minimum coverage constraint for every method
TOP_ENRICHED = 8                  # SNN BUS: conditions per mini-cluster (raised to the depth if the depth is larger)
BART_MIN_TIME = 60                # Bart's exact BUS gets as much time as BS w=50 took on the same task, at least this
                                  # many seconds (Engelen §4.3.4: time budget); unfinished runs are reported
POOL_W = 10                       # beam width of both BUS→BS pool ensembles
METHODS_B = {methods!r}
RESULTS_DIR = f"results_binary_quick/{{DATASET}}" if QUICK else f"results_binary/{{DATASET}}"
os.makedirs(RESULTS_DIR, exist_ok=True)
N_JOBS = min(N_JOBS, 7)
BATCH = 2 * N_JOBS
print(DATASET, "| methods:", METHODS_B)
'''

CORE = r'''
# ================= binary study: generator, seeds, methods =================
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def binary_uneven(rng, k):
    """k binary attributes; attribute j is 1 for a share p_j ~ U(P_RANGE) of the records (Bart's generator has p = 0.5)."""
    p = rng.uniform(P_RANGE[0], P_RANGE[1], size=k)
    return (rng.random((N, k)) < p).astype(np.int8)


def value_freqs(A_):
    return [np.bincount(A_[:, j], minlength=2) / len(A_) for j in range(A_.shape[1])]


def n_stores_of(mask):
    return len(np.unique(store_codes[mask]))


def sample_true_description(rng, A_, freqs, lits, tries=5000):
    """lits random attributes, each with a value drawn in proportion to its frequency; resampled until the subgroup
    covers COVERAGE_RANGE of the records and at least MIN_STORES stores."""
    for _ in range(tries):
        attrs = rng.choice(A_.shape[1], lits, replace=False)
        desc = tuple(sorted((int(j), int(rng.choice(2, p=freqs[j]))) for j in attrs))
        mask = mask_of(A_, desc)
        if COVERAGE_RANGE[0] <= mask.mean() <= COVERAGE_RANGE[1] and n_stores_of(mask) >= MIN_STORES:
            return desc, mask
    return None, None


def block_means_n(rel, n_blocks=N_BLOCKS):
    post = rel[:, BASE_LEN:]
    k = max(1, post.shape[1] // n_blocks)
    m = (post.shape[1] // k) * k
    return post[:, :m].reshape(len(rel), -1, k).mean(axis=2)


def pairwise_sqeuclidean(X, block=512):
    """Full N x N squared Euclidean distance matrix (Engelen eq. 2.15), row blocks in parallel."""
    starts = list(range(0, len(X), block))
    parts = Parallel(n_jobs=N_JOBS)(delayed(cdist)(X[s:s + block], X, "sqeuclidean") for s in starts)
    D = np.empty((len(X), len(X)), dtype=np.float32 if len(X) > 6000 else np.float64)
    for s, part in zip(starts, parts):
        D[s:s + len(part)] = part
    return D


def update_rows_sq(D, X, rows):
    D = D.copy()
    if len(rows):
        new = cdist(X[rows], X, "sqeuclidean").astype(D.dtype)
        D[rows, :] = new
        D[:, rows] = new.T
    return D


# ---------------- Bart's BUS (Engelen Alg. 2/3), exact, with a time budget
def bart_exact(pairs, A_, d, phi, time_limit):
    """Every combination (up to depth d) of the conditions a pair shares, level by level. A combination is skipped
    only if a sub-description failed the coverage constraint or its optimistic estimate was below the current q-th
    quality: both rules are lossless, so the top-q equals that of the full Alg. 3 if the run finishes."""
    spaces = [[(int(j), int(v)) for j, v in shared_conditions(list(pr), A_)] for pr in pairs]
    rs, stats, t_end = ResultSet(Q), Counter(), time.time() + time_limit
    fronts, prev_alive = [[()] for _ in spaces], {()}
    for size in range(1, d + 1):
        seen, cur_alive, new_fronts = {}, set(), []
        for S, front in zip(spaces, fronts):
            nf = []
            for I in front:
                for k in range(I[-1] + 1 if I else 0, len(S)):
                    J = I + (k,)
                    desc = tuple(S[i] for i in J)
                    if desc not in seen:
                        ok = size == 1 or all(desc[:m] + desc[m + 1:] in prev_alive for m in range(size))
                        if ok:
                            mask = mask_of(A_, desc)
                            if not phi.expandable(mask):
                                ok = False
                            else:
                                quality = phi(mask)
                                if phi.satisfies_constraints(mask):
                                    rs.insert(quality, desc, mask)
                                if oe_below_threshold(phi, mask, rs):
                                    ok = False
                        seen[desc] = ok
                        if ok:
                            cur_alive.add(desc)
                        if time.time() > t_end:
                            return rs.sorted(), stats, False, spaces
                    if seen[desc]:
                        nf.append(J)
            new_fronts.append(nf)
        stats["levels_done"] = size
        fronts, prev_alive = new_fronts, cur_alive
    return rs.sorted(), stats, True, spaces


# ---------------- fixed BUS: SNN components on the scale-free, time-blocked target
def snn_components_exp(D, max_size, k=KNN, snn_min=SNN_MIN, u=U_SEEDS):
    knn = knn_lists(D, k=k)
    sets = [set(r) for r in knn.tolist()]
    edges = [(i, j) for i, row in enumerate(knn.tolist()) for j in row
             if j > i and i in sets[j] and len(sets[i] & sets[j]) >= snn_min]
    if not edges:
        return []
    i = np.array([e[0] for e in edges])
    j = np.array([e[1] for e in edges])
    _, labels = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(D), len(D))), directed=False)
    comps = [c for c in pd.Series(np.arange(len(D))).groupby(labels).apply(list) if len(c) >= Z_COMP_MIN]
    out = []
    for c in comps:
        members = np.unique(np.concatenate([np.asarray(c), knn[list(c)].ravel()]))
        if len(members) <= max_size:
            out.append(tuple(members.tolist()))
    return sorted(set(out), key=len, reverse=True)[:u]


def enriched(mc, A_, freqs, top, z_min=2.0):
    """Conditions shared by >= MAJORITY of the mini-cluster, ranked by binomial enrichment z (z >= z_min), at most top."""
    sub = A_[list(mc)]
    n = len(mc)
    conds = []
    for j in range(A_.shape[1]):
        ones = int(sub[:, j].sum())
        for v, c in ((1, ones), (0, n - ones)):
            s, base = c / n, freqs[j][v]
            if s >= MAJORITY and 0 < base < 1:
                z = (s - base) * np.sqrt(n) / np.sqrt(base * (1 - base))
                if z >= z_min:
                    conds.append((z, (j, v)))
    conds.sort(reverse=True)
    return [c for _, c in conds[:top]]


def parse_method(name):
    if name.startswith("BS"):
        return {"kind": "bs", "w": int(name.split("w=")[1]), "local": False}
    seeds = "snn" if "snn" in name else ("alpha" if ("α" in name or "pool" in name) else "closest")
    kind = "pool" if "pool" in name else "bus"
    return {"kind": kind, "seeds": seeds, "local": name.endswith("+local"),
            "base": name.replace(" +local", "") if name.endswith("+local") else None}


METHOD_SPECS = {m: parse_method(m) for m in METHODS_B}
BUDGETS_B = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
SECONDS_B = [1, 2, 5, 10, 30, 60, 120]


def _binary_job(spec, excess, level, store_codes_, D_blocks, D_bart):
    rng = np.random.default_rng(spec["seed"])
    lits = spec["lits"]
    d = lits                                       # every method searches to the true depth
    A_ = binary_uneven(rng, spec["k"])
    freqs = value_freqs(A_)
    planted_desc, planted = sample_true_description(rng, A_, freqs, lits)
    if planted_desc is None:
        return []
    rows_ = np.flatnonzero(planted)
    keep = rows_[rng.random(len(rows_)) >= spec["outliers"]]
    exc = excess.copy()
    exc[keep] += spec["delta"] * POST[None, :] * level[keep, None]
    rel = np.clip(exc / level[:, None], -3, 3)
    max_size = int(MAX_COV * len(level))
    pset = set(planted_desc)

    cache = {}

    def seeds(kind):
        """Mini-clusters + the time to build them (distance update included)."""
        if kind not in cache:
            t = time.time()
            if kind == "snn":
                D = update_rows(D_blocks, block_means_n(rel), keep)
                mcs = snn_components_exp(D, max_size)
            else:
                if "D_bart" not in cache:
                    cache["D_bart"] = update_rows_sq(D_bart, exc, keep)
                mcs = top_pairs(cache["D_bart"], U_SEEDS, kind)
            cache[kind] = (mcs, time.time() - t)
        return cache[kind]

    def make_row(name, phi, top, t0, offset, info):
        trace = [(tt - t0 + offset, q_, jac, valid) for tt, q_, jac, valid in phi.trace]
        top_jac = [np.count_nonzero(m & planted) / np.count_nonzero(m | planted) for _, _, m in top]
        ranks = [r + 1 for r, jac in enumerate(top_jac) if jac >= HIT_JACCARD]
        row = {k_: v_ for k_, v_ in spec.items()}
        row.update({"method": name, "depth_searched": d, "planted_n": int(planted.sum()), "coverage": planted.mean(),
                    "evaluated_hit": any(valid and jac >= HIT_JACCARD for _, _, jac, valid in trace),
                    "rank": ranks[0] if ranks else np.nan, "evaluated": len(trace),
                    "runtime_s": time.time() - t0 + offset, "seed_time_s": offset, **info})
        for b in BUDGETS_B:
            row[f"rank_at_{b}"] = _rank_of_planted(trace[:b])
        for s_ in SECONDS_B:
            row[f"rank_at_{s_}s"] = _rank_of_planted([e for e in trace if e[0] <= s_])
        return row

    out, state, bs50 = [], {}, {"runtime": 0.0}
    for name, s in METHOD_SPECS.items():
        if s["local"]:                              # continue from the base method's run (same quality object)
            if s["base"] not in state:
                raise ValueError(f"{name} must come right after {s['base']}")
            phi, top, t0, offset, info = state[s["base"]]
            rs = ResultSet(Q)
            for q_, d_, m_ in top:
                rs.insert(q_, d_, m_)
            stats = Counter()
            top = local_search_d(A_, phi, top, rs, stats, d)
            out.append(make_row(name, phi, top, t0, offset, dict(info, local_evaluated=stats["local_evaluated"])))
            continue
        phi = MonetaryQuality(exc, level, store_codes_, relative=True, min_cov=MIN_COV_B)
        phi.track, phi.trace = planted, []
        info, offset = {}, 0.0
        if s["kind"] == "bs":
            t0 = time.time()
            top, _ = beam_search(A_, phi, w=s["w"], d=d, prune=True)
            if s["w"] == 50:
                bs50["runtime"] = time.time() - t0
        else:
            mcs, offset = seeds(s["seeds"])
            t0 = time.time()
            if s["seeds"] == "snn":
                baskets = [enriched(mc, A_, freqs, max(TOP_ENRICHED, d)) for mc in mcs]
            else:
                baskets = [[(int(j), int(v)) for j, v in shared_conditions(list(mc), A_)] for mc in mcs]
            info = {"mini_clusters": len(mcs), "mean_mc_size": float(np.mean([len(m) for m in mcs])) if mcs else 0.0,
                    "best_mc_purity": max((planted[list(m)].mean() for m in mcs), default=0.0),
                    "pair_size_pct": float(np.mean([LEVEL_PCT[list(m)].mean() for m in mcs])) if mcs else np.nan,
                    "planted_conds_in_basket": max((len(set(b) & pset) for b in baskets), default=0),
                    "planted_in_space": any(pset <= set(b) for b in baskets),
                    "basket_size": float(np.mean([len(b) for b in baskets])) if baskets else 0.0}
            if s["kind"] == "pool":
                votes = Counter(c for b in baskets for c in b)
                top, _ = beam_search(A_, phi, w=POOL_W, d=d, prune=True,
                                     pool=[c for c, _ in votes.most_common(POOL_SIZE)])
            elif s["seeds"] == "snn":
                cands = set()
                for b in baskets:
                    for size in range(1, min(d, len(b)) + 1):
                        cands.update(tuple(sorted(c)) for c in itertools.combinations(b, size))
                info["candidates"] = len(cands)
                top, _ = evaluate_candidates(A_, phi, cands, prune=True)
            else:
                budget = max(BART_MIN_TIME, bs50["runtime"])
                top, st, finished, _ = bart_exact(mcs, A_, d, phi, budget)
                info.update(finished=finished, levels_done=st["levels_done"], time_budget_s=budget)
        state[name] = (phi, top, t0, offset, info)
        out.append(make_row(name, phi, top, t0, offset, info))
    return out
'''

RUN = r'''
# ---- calibration (deterministic): null spread of random true-subgroup-like descriptions, per depth
rng = np.random.default_rng(SEED + 11)
SIGMA_LITS = {}
for lits in LITS_B:
    rates = []
    while len(rates) < 200:
        A_ = binary_uneven(rng, 100)
        desc, mask = sample_true_description(rng, A_, value_freqs(A_), lits)
        if desc is not None:
            rates.append(excess[mask][:, POST].sum() / (level[mask].sum() * POST.sum()))
    SIGMA_LITS[lits] = float(np.std(rates))
print("null spread per depth:", {k: round(v, 4) for k, v in SIGMA_LITS.items()})

LEVEL_PCT = pd.Series(level).rank(pct=True).to_numpy()      # size of a series (0 = smallest, 1 = largest)
D_blocks = pairwise_cityblock(block_means_n(rel_dev))      # fixed BUS: scale-free, time-blocked target
D_bart = pairwise_sqeuclidean(excess)                      # Bart's BUS: squared Euclidean on the target as-is

specs = []
for k in K_ATTRS:
    for lits in LITS_B:
        for snr, outliers in GRID:
            for rep in range(REPS_B):
                specs.append({"dataset": DATASET, "k": k, "lits": lits, "snr": snr, "outliers": outliers, "rep": rep,
                              "delta": snr * SIGMA_LITS[lits], "seed": int(rng.integers(1 << 31))})

out_csv, progress = f"{RESULTS_DIR}/tasks.csv", f"{RESULTS_DIR}/progress.txt"
skipped_file = f"{RESULTS_DIR}/skipped_seeds.txt"
done = set(pd.read_csv(out_csv)["seed"]) if os.path.exists(out_csv) else set()
skipped = set(int(x) for x in open(skipped_file).read().split()) if os.path.exists(skipped_file) else set()
todo = sorted([s for s in specs if s["seed"] not in done and s["seed"] not in skipped], key=lambda s: (s["rep"], s["lits"], s["k"]))
print(f"{DATASET}: {len(specs)} tasks x {len(METHODS_B)} methods; {len(done) + len(skipped)} already done, {len(todo)} to run")
start = time.time()
for b in range(0, len(todo), BATCH):
    batch = todo[b:b + BATCH]
    parts = Parallel(n_jobs=N_JOBS)(delayed(_binary_job)(s, excess, level, store_codes, D_blocks, D_bart) for s in batch)
    rows = [r for p in parts for r in p]
    if rows:
        new = pd.DataFrame(rows)
        if os.path.exists(out_csv):            # same columns, same order as the file
            cols = list(pd.read_csv(out_csv, nrows=0).columns)
            extra = [c for c in new.columns if c not in cols]
            assert not extra, f"new columns {extra} not in {out_csv}"
            new.reindex(columns=cols).to_csv(out_csv, mode="a", header=False, index=False)
        else:
            new.to_csv(out_csv, index=False)
    with open(skipped_file, "a") as f:
        for s, p in zip(batch, parts):
            if not p:
                f.write(f"{s['seed']}\n")
    elapsed = time.time() - start
    left = elapsed / (b + len(batch)) * (len(todo) - b - len(batch))
    with open(progress, "w") as f:
        f.write(f"{DATASET}: {len(done) + len(skipped) + b + len(batch)} / {len(specs)} tasks, "
                f"{elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left\n")
print(f"done in {time.time() - start:.0f}s")
'''

SUMMARY = r'''
# ---- results: every method vs BS w=10 and vs BUS snn on the same tasks (exact McNemar test)
from scipy.stats import binomtest

res = pd.read_csv(f"{RESULTS_DIR}/tasks.csv")
res["found"] = res["rank"] <= Q
methods = [m for m in METHODS_B if m in set(res.method)]
key = ["k", "lits", "snr", "outliers", "rep"]
print(f"{DATASET}: {res.seed.nunique()} tasks built (of {len(specs)}; the rest had no valid true subgroup)")
print("Average per task")
display(res.groupby("method")[["found", "evaluated_hit", "evaluated", "runtime_s"]].mean().reindex(methods).round(3))
print("Share found, by depth (rows) and method (columns)")
display(res.pivot_table(index="lits", columns="method", values="found")[methods].round(3))
print("Share found, by number of binary attributes k")
display(res.pivot_table(index="k", columns="method", values="found")[methods].round(3))
if "finished" in res:
    print("Bart's exact BUS: share of runs finished within the time limit, by k and depth")
    display(res[res.method.isin(["BUS", "BUS-α"])].pivot_table(index="k", columns="lits", values="finished").round(2))

w = res.pivot_table(index=key, columns="method", values="found").astype(bool)
families = [([("all", w)], 0.05),
            ([(f"lits={l}", w.xs(l, level="lits", drop_level=False)) for l in sorted(res.lits.unique())], 0.05 / res.lits.nunique()),
            ([(f"k={k}", w.xs(k, level="k", drop_level=False)) for k in sorted(res.k.unique())], 0.05 / res.k.nunique())]
rows = []
for ref in ["BS w=10", "BUS snn"]:
    for groups, alpha in families:
        for g, sub in groups:
            for m in [x for x in methods if x != ref]:
                wins, losses = int((sub[m] & ~sub[ref]).sum()), int((~sub[m] & sub[ref]).sum())
                p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
                verdict = (f"better than {ref}" if wins > losses else f"worse than {ref}") if p < alpha else \
                          ("suggestive (not after correction)" if p < 0.05 else "no significant difference")
                rows.append({"vs": ref, "group": g, "method": m, "tasks": len(sub), "found": sub[m].mean(),
                             "found_ref": sub[ref].mean(), "wins": wins, "losses": losses, "p": p, "alpha": alpha,
                             "verdict": verdict})
paired = pd.DataFrame(rows)
paired.to_csv(f"{RESULTS_DIR}/paired.csv", index=False)
print("Paired comparison (wins = only this method found it, losses = only the reference found it)")
display(paired[paired.vs == "BS w=10"].round(4))
display(paired[paired.vs == "BUS snn"].round(4))
'''


def main():
    import nbformat

    QUICK = sys.argv[1] == "quick"
    ONLY = set(sys.argv[2:])

    def load(path):
        return nbformat.read(path, as_version=4)


    loaders = {"walmart": load("walmart_subtle.ipynb"), "favorita": load("favorita_subtle.ipynb")}
    for ds in DATASETS:
        if ONLY and ds not in ONLY:
            continue
        nb = load("m5_deep_v5.ipynb")
        for c in nb.cells:
            if c.cell_type == "code":
                c.outputs, c.execution_count = [], None
        if ds != "m5":
            nb.cells[2].source = loaders[ds].cells[2].source        # dataset settings
            nb.cells[6].source = loaders[ds].cells[3].source        # data loading
        nb.cells[2].source = nb.cells[2].source.replace("QUICK = False", f"QUICK = {QUICK}")
        assert f"QUICK = {QUICK}" in nb.cells[2].source
        nb.cells[0].source = (f"# Binary semi-synthetic study · {ds}\n"
                              "Same algorithms, quality measure (M2), constraints and task design on M5, Walmart and Favorita.\n\n"
                              "- Descriptors: k ∈ {20, 50, 100, 400} binary attributes; attribute j is 1 for a share p_j ~ U(0.5, 0.9)\n"
                              "  of the records (uneven binary, so deep descriptions keep enough records).\n"
                              "- True subgroup: 2, 4, 6, 8 or 10 conditions, 1–15% coverage, ≥ 2 stores; level step δ = snr × σ\n"
                              "  (σ = null spread of random descriptions of that depth), snr 8 / 16, outliers 0 / 20%, 10 repetitions.\n"
                              f"- Methods: {', '.join(METHODS)}.\n"
                              "- Bart's BUS (closest / α pairs) runs exactly, with the time BS w=50 took on the same task (at least 60 s); unfinished runs are reported.\n"
                              "Resumable: progress in results_binary/<dataset>/progress.txt; rerunning skips finished tasks.")
        settings = SETTINGS.format(methods=METHODS)
        nb.cells = nb.cells[:14] + [nbformat.v4.new_code_cell(x.strip()) for x in (settings, CORE, RUN, SUMMARY)]
        nbformat.write(nb, f"binary_{ds}.ipynb")
        print("wrote", f"binary_{ds}.ipynb")


if __name__ == "__main__":
    main()
