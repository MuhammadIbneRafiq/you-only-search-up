"""Weekend suite: builds suite_<run>.ipynb notebooks from m5_cat_v7.ipynb (+ the Walmart / Favorita loaders).
Usage: python make_suite.py <quick|full> [run ...]"""
import json
import sys

import nbformat

QUICK = sys.argv[1] == "quick"
ONLY = set(sys.argv[2:])

FULL_METHODS = ["BS w=10", "BS w=50", "BUS", "BUS +local", "pool", "itemsets s2", "itemsets s2 +guided"]
RUNS = {
    "main": dict(target="m5", generator="cat", reps=30, depth="true",
                 methods=["BS w=5", "BS w=10", "BS w=20", "BS w=50", "BUS", "BUS +local", "pool",
                          "itemsets s1", "itemsets s2", "itemsets s1 +guided", "itemsets s2 +guided"],
                 why="All methods in one run on identical tasks (main comparison incl. the itemset / co-occurrence methods)."),
    "seeds": dict(target="m5", generator="cat", reps=15, depth="true", needs_full_distances=True,
                  methods=["BS w=10", "BUS seed=closest-$", "BUS seed=closest-rel", "BUS seed=closest-blocks",
                           "BUS seed=snn-pairs", "BUS seed=snn-raw", "BUS", "BUS describe=share", "BUS describe=all"],
                  why="Ablation of the fix: seed type (old dollar pairs → scale-free → time blocks → SNN pairs → SNN components → expanded) and the description step."),
    "depth6": dict(target="m5", generator="cat", reps=15, depth=6,
                   methods=["BS w=10", "BS w=20", "BUS", "BUS +local", "pool", "itemsets s2", "itemsets s2 +guided"],
                   why="Unknown depth: every method searches up to depth 6, whatever the true depth."),
    "hier": dict(target="m5", generator="hier", reps=15, depth="true", methods=FULL_METHODS,
                 why="Correlated / hierarchical descriptors (child attribute follows its parent with probability 0.8)."),
    "walmart": dict(target="walmart", generator="cat", reps=15, depth="true", methods=FULL_METHODS,
                    why="Same design on the Walmart target (2,566 series x 143 weeks)."),
    "favorita": dict(target="favorita", generator="cat", reps=15, depth="true", methods=FULL_METHODS,
                     why="Same design on the Favorita target (1,100 series x 910 days)."),
    "effect": dict(target="m5", generator="cat", reps=15, depth="true",
                   grid=[(2, 0.0), (2, 0.2), (4, 0.0), (4, 0.2), (8, 0.5), (16, 0.5)],
                   methods=["BS w=10", "BUS", "BUS +local", "pool", "itemsets s2 +guided"],
                   why="Weaker effects (2σ, 4σ) and heavy contamination (50% of the true subgroup unchanged, as in Bart)."),
    "sensitivity": dict(target="m5", generator="cat", reps=15, depth="true",
                        methods=["BS w=10", "BUS", "BUS k=5", "BUS k=20", "BUS snn=3", "BUS snn=7", "BUS u=50", "BUS u=400",
                                 "BUS top=4", "BUS top=12", "BUS z=1", "BUS z=3", "BUS maj=0.5", "BUS maj=0.7",
                                 "BUS blocks=12", "BUS blocks=108"],
                        why="One-at-a-time sensitivity of BUS's settings around the defaults (k=10, snn=5, u=200, top=8, z=2, maj=0.6, blocks=36)."),
    "beam100": dict(target="m5", generator="cat", reps=10, depth="true", methods=["BS w=10", "BS w=100", "BUS"],
                    why="Does a much wider beam (w=100) close the gap?"),
    "original_exact": dict(target="m5", generator="cat", reps=30, depth="true", needs_original=True,
                           methods=["BS w=10", "BUS", "BUS-orig dist=$ budget=none", "BUS-orig dist=rel budget=none",
                                    "BUS-orig dist=rel pairs=alpha budget=none"],
                           why="Bart's original BUS without any candidate cap: every combination of the shared conditions up to the "
                               "depth, enumerated level-wise with the same lossless pruning every method gets (coverage constraint and "
                               "exact optimistic estimate), so the top-20 equals that of an uncapped, unpruned run."),
    "original": dict(target="m5", generator="cat", reps=30, depth="true", needs_original=True,
                     methods=["BS w=10", "BUS", "BUS-orig dist=$", "BUS-orig dist=$ pairs=alpha", "BUS-orig dist=rel",
                              "BUS-orig dist=rel pairs=alpha"],
                     why="Bart's original BUS (Alg. 2/3: the u=200 closest pairs in squared Euclidean distance on the full target, "
                         "every condition shared by both members, every combination up to the depth; also with the alpha-matrix pairs, "
                         "§4.3.2), on the dollar target and on the scale-free target, against fixed BUS and BS w=10 on the same tasks."),
}

SUITE_SETTINGS = '''
# ================= weekend suite: run settings =================
RUN = "{run}"
GENERATOR = "{generator}"          # "cat" = independent PoS-like categorical; "hier" = parent/child correlated
HIER_RHO = 0.8                     # hier: probability that a child attribute follows its parent
K_ATTRS = [200, 800]               # target number of conditions (attribute = value)
LITS_V5 = [3, 4] if QUICK else [3, 4, 5, 6]
GRID = {grid}                      # (snr, outliers) combinations
if QUICK:
    GRID = GRID[:1]
REPS_V5 = 1 if QUICK else {reps}
DEPTH = {depth!r}                  # "true" = every method gets the true depth; an int = fixed maximum depth for all
SUITE_METHODS = {methods!r}
NEEDS_FULL_DISTANCES = {needs}     # only the seed ablation uses the full-T dollar / relative distances
NEEDS_ORIGINAL = {needs_orig}      # Bart's original BUS: squared Euclidean distances on the full target
EXACT_TIME_LIMIT = 1800             # uncapped original BUS: seconds per run before it is reported as unfinished
CAND_BUDGET = 50000                # original BUS: at most this many candidates (all 1-condition ones first, then 2, ...)
RESULTS_DIR = f"results_suite_quick/{{RUN}}" if QUICK else f"results_suite/{{RUN}}"
os.makedirs(RESULTS_DIR, exist_ok=True)
N_JOBS = min(N_JOBS, 7)
BATCH = 2 * N_JOBS
print(RUN, "-", {why!r})
'''

SUITE_CORE = r'''
# ================= weekend suite: generators, seeds, descriptions, methods =================
def zipf_column(rng, c):
    p = 1.0 / np.arange(1, c + 1) ** ZIPF_S
    return rng.choice(c, size=N, p=p / p.sum())


def hier_descriptors(rng, n_conditions, rho=HIER_RHO):
    """Correlated descriptors: groups of a parent attribute and 1-2 children; a child takes the value mapped from its
    parent with probability rho, otherwise a Zipf value (like category -> department, or price bin -> price level)."""
    cols, total = [], 0
    while total < n_conditions:
        c_parent = int(rng.choice(CARDINALITIES))
        parent = zipf_column(rng, c_parent)
        cols.append(parent)
        total += c_parent
        for _ in range(int(rng.integers(1, 3))):
            if total >= n_conditions:
                break
            c_child = int(rng.choice([c for c in CARDINALITIES if c >= c_parent]))
            mapping = rng.integers(c_child, size=c_parent)
            child = np.where(rng.random(N) < rho, mapping[parent], zipf_column(rng, c_child))
            cols.append(child)
            total += c_child
    return np.stack(cols, axis=1).astype(np.int16)


def make_descriptors(rng, k):
    return hier_descriptors(rng, k) if GENERATOR == "hier" else categorical_descriptors(rng, k)


def block_means_n(rel, n_blocks):
    post = rel[:, BASE_LEN:]
    k = max(1, post.shape[1] // n_blocks)
    m = (post.shape[1] // k) * k
    return post[:, :m].reshape(len(rel), -1, k).mean(axis=2)


def snn_edges_k(D, k):
    knn = knn_lists(D, k=k)
    sets = [set(r) for r in knn.tolist()]
    edges = []
    for i, row in enumerate(knn.tolist()):
        for j in row:
            if j > i and i in sets[j]:
                edges.append((len(sets[i] & sets[j]), float(D[i, j]), i, j))
    return knn, edges


def seeds_of(kind, D, k, snn_min, u, max_size):
    """Mini-clusters. Pairs: closest pairs in D. snn-pairs: mutual neighbours sharing the most neighbours.
    snn-raw: connected components of the SNN graph. snn-exp: components expanded with their members' neighbours."""
    if kind in ("closest-$", "closest-rel", "closest-blocks"):
        return top_pairs(D, u, "closest")
    knn, edges = snn_edges_k(D, k)
    if kind == "snn-pairs":
        return [(i, j) for _, _, i, j in sorted(edges, key=lambda e: (-e[0], e[1]))[:u]]
    strong = [e for e in edges if e[0] >= snn_min]
    if not strong:
        return []
    i = np.array([e[2] for e in strong])
    j = np.array([e[3] for e in strong])
    _, labels = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(D), len(D))), directed=False)
    comps = [c for c in pd.Series(np.arange(len(D))).groupby(labels).apply(list) if len(c) >= Z_COMP_MIN]
    if kind == "snn-raw":
        out = [tuple(c) for c in comps if len(c) <= max_size]
    else:
        out = []
        for c in comps:
            members = np.unique(np.concatenate([np.asarray(c), knn[list(c)].ravel()]))
            if len(members) <= max_size:
                out.append(tuple(members.tolist()))
    return sorted(set(out), key=len, reverse=True)[:u]


def describe_mc(mc, A_, freqs, rule, z_min, majority, top):
    """Basket of a mini-cluster. z: shared by >= majority, ranked by binomial enrichment z (z >= z_min).
    share: shared by >= majority, ranked by share only (no rarity). all: shared by every member (Bart Alg. 3),
    rarest first. At most `top` conditions."""
    sub = A_[list(mc)]
    n = len(mc)
    conds = []
    for j in range(A_.shape[1]):
        values, counts = np.unique(sub[:, j], return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            s, base = c / n, freqs[j][v]
            if not 0 < base < 1:
                continue
            if rule == "all":
                if c == n:
                    conds.append((-base, (j, v)))
            elif s >= majority:
                if rule == "share":
                    conds.append((s, (j, v)))
                else:
                    z = (s - base) * np.sqrt(n) / np.sqrt(base * (1 - base))
                    if z >= z_min:
                        conds.append((z, (j, v)))
    conds.sort(reverse=True)
    return [c for _, c in conds[:top]]


def pairwise_sqeuclidean(X, block=512):
    """Full N x N squared Euclidean distance matrix (Bart eq. 2.15), row blocks computed in parallel."""
    N_ = len(X)
    starts = list(range(0, N_, block))
    parts = Parallel(n_jobs=N_JOBS)(delayed(cdist)(X[s:s + block], X, "sqeuclidean") for s in starts)
    D = np.empty((N_, N_), dtype=np.float32 if N_ > 6000 else np.float64)
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


def bart_candidates(pairs, A_, d, budget):
    """Bart Alg. 3: the conditions shared by both members of a pair; every combination up to depth d.
    Enumerated by size (every 1-condition description first, then 2, ...) until the budget is reached."""
    spaces = [[(int(j), int(v)) for j, v in shared_conditions(list(pr), A_)] for pr in pairs]
    cands = set()
    for size in range(1, d + 1):
        for conds in spaces:
            for c in itertools.combinations(conds, size):
                cands.add(c)
                if len(cands) >= budget:
                    return cands, spaces, True
    return cands, spaces, False


def bart_exact(pairs, A_, d, phi, time_limit):
    """Bart Alg. 3 without a cap: every combination (up to depth d) of the conditions a pair shares, level by level.
    A combination is skipped only when one of its sub-descriptions failed the coverage constraint or had an optimistic
    estimate below the current q-th best quality. Both rules are lossless (coverage is anti-monotone, the estimate
    bounds every refinement and the threshold only rises), so the top-q equals that of the full, unpruned Alg. 3."""
    spaces = [[(int(j), int(v)) for j, v in shared_conditions(list(pr), A_)] for pr in pairs]
    rs, stats, t_end = ResultSet(Q), Counter(), time.time() + time_limit
    fronts, prev_alive = [[()] for _ in spaces], {()}
    stats["levels_done"] = 0
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
                            stats["generated"] += 1
                            mask = mask_of(A_, desc)
                            if not phi.expandable(mask):
                                ok = False
                            else:
                                quality = phi(mask)
                                stats["evaluated"] += 1
                                if phi.satisfies_constraints(mask):
                                    rs.insert(quality, desc, mask)
                                if oe_below_threshold(phi, mask, rs):
                                    ok = False
                        else:
                            stats["pruned_superset"] += 1
                        seen[desc] = ok
                        if ok:
                            cur_alive.add(desc)
                        if time.time() > t_end:
                            return rs.sorted(), stats, False
                    if seen[desc]:
                        nf.append(J)
            new_fronts.append(nf)
        stats[f"alive_level_{size}"] = len(cur_alive)
        stats["levels_done"] = size
        fronts, prev_alive = new_fronts, cur_alive
    return rs.sorted(), stats, True


DEFAULTS = dict(seed="snn-exp", describe="z", k=KNN, snn=SNN_MIN, u=U_SEEDS, top=TOP_ENRICHED, z=2.0, maj=MAJORITY,
                blocks=N_BLOCKS)


def parse_method(name):
    """'BS w=10' | 'BUS[ key=value ...][ +local]' | 'pool' | 'itemsets sK[ +guided]'."""
    if name.startswith("BS"):
        return {"kind": "bs", "w": int(name.split("w=")[1])}
    if name.startswith("BUS-orig"):
        spec = {"kind": "orig", "dist": "$", "pairs": "closest", "u": U_SEEDS}
        for token in name.split()[1:]:
            key, value = token.split("=")
            spec[key] = int(value) if key == "u" else value
        return spec
    spec = dict(DEFAULTS)
    if name.startswith("pool"):
        spec["kind"] = "pool"
    elif name.startswith("itemsets"):
        spec.update(kind="itemsets", minsup=int(name.split()[1][1:]), guided=name.endswith("+guided"))
    else:
        spec.update(kind="bus", local=name.endswith("+local"))
        for token in name.split()[1:]:
            if "=" in token:
                key, value = token.split("=")
                spec[key] = value if key in ("seed", "describe") else (float(value) if key in ("z", "maj") else int(value))
    return spec


METHOD_SPECS = {m: parse_method(m) for m in SUITE_METHODS}
BLOCK_SIZES = sorted({s["blocks"] for s in METHOD_SPECS.values() if s["kind"] not in ("bs", "orig")} | {N_BLOCKS})


def _suite_job(spec, excess, level, store_codes, D_blocks_by, D_money_full, D_shape_full, method_specs,
               D_orig_money=None, D_orig_rel=None):
    rng = np.random.default_rng(spec["seed"])
    lits = spec["lits"]
    d = lits if DEPTH == "true" else int(DEPTH)
    A_ = make_descriptors(rng, spec["k"])
    freqs = value_freqs(A_)
    planted_desc, planted = sample_description_cat(rng, A_, freqs, lits, store_codes)
    if planted_desc is None:
        return []
    rows_ = np.flatnonzero(planted)
    keep = rows_[rng.random(len(rows_)) >= spec["outliers"]]
    exc = excess.copy()
    exc[keep] += spec["delta"] * POST[None, :] * level[keep, None]
    rel = np.clip(exc / level[:, None], -3, 3)
    max_size = int(MAX_COV * len(level))
    LEVEL_PCT = pd.Series(level).rank(pct=True).to_numpy()   # size of a series (0 = smallest, 1 = largest)

    dist_cache, seed_cache, basket_cache = {}, {}, {}

    def distances(kind, blocks):
        key = (kind, blocks)
        if key not in dist_cache:
            t = time.time()
            if kind == "closest-$":
                D = update_rows(D_money_full, exc, keep)
            elif kind == "closest-rel":
                D = update_rows(D_shape_full, rel, keep)
            else:
                D = update_rows(D_blocks_by[blocks], block_means_n(rel, blocks), keep)
            dist_cache[key] = (D, time.time() - t)
        return dist_cache[key]

    def baskets(s):
        key = (s["seed"], s["k"], s["snn"], s["u"], s["blocks"], s["describe"], s["z"], s["maj"], s["top"])
        if key not in basket_cache:
            dkind = s["seed"] if s["seed"] in ("closest-$", "closest-rel") else "blocks"
            D, dtime = distances(dkind, s["blocks"])
            t = time.time()
            mcs = seeds_of(s["seed"], D, s["k"], s["snn"], s["u"], max_size)
            b = [describe_mc(mc, A_, freqs, s["describe"], s["z"], s["maj"], s["top"]) for mc in mcs]
            purity = [planted[list(mc)].mean() for mc in mcs]
            info = {"mini_clusters": len(mcs), "mean_mc_size": float(np.mean([len(m) for m in mcs])) if mcs else 0.0,
                    "best_mc_purity": max(purity, default=0.0),
                    "planted_conds_in_basket": max((len(set(x) & set(planted_desc)) for x in b), default=0)}
            basket_cache[key] = (b, time.time() - t + dtime, info)
        return basket_cache[key]

    out = []
    for name, s in method_specs.items():
        phi = MonetaryQuality(exc, level, store_codes, relative=True, min_cov=MIN_COV_V5)
        phi.track, phi.trace = planted, []
        stats, offset, info = Counter(), 0.0, {}
        t0 = time.time()
        if s["kind"] == "bs":
            top, stats = beam_search(A_, phi, w=s["w"], d=d, prune=True)
        elif s["kind"] == "orig":
            okey = ("orig", s["dist"])
            if okey not in dist_cache:
                t = time.time()
                D = update_rows_sq(D_orig_money, exc, keep) if s["dist"] == "$" else update_rows_sq(D_orig_rel, rel, keep)
                dist_cache[okey] = (D, time.time() - t)
            D, dtime = dist_cache[okey]
            t = time.time()
            pairs = top_pairs(D, s["u"], s["pairs"])
            if s.get("budget") == "none":
                offset = dtime + time.time() - t
                t0 = time.time()
                top, stats, finished = bart_exact(pairs, A_, d, phi, EXACT_TIME_LIMIT)
                spaces = [[(int(j), int(v)) for j, v in shared_conditions(list(pr), A_)] for pr in pairs]
                pset = set(planted_desc)
                info = {"mini_clusters": len(pairs), "mean_mc_size": 2.0, "finished": finished,
                        "levels_done": stats["levels_done"], "generated": stats["generated"],
                        "best_mc_purity": max((planted[list(pr)].mean() for pr in pairs), default=0.0),
                        "planted_conds_in_basket": max((len(set(x) & pset) for x in spaces), default=0),
                        "planted_in_space": any(pset <= set(x) for x in spaces),
                        "shared_per_pair": float(np.mean([len(x) for x in spaces])) if spaces else 0.0,
                        "pair_level_pct": float(np.mean([LEVEL_PCT[list(pr)].mean() for pr in pairs])) if pairs else np.nan,
                        **{f"alive_level_{i}": stats.get(f"alive_level_{i}", np.nan) for i in range(1, 7)}}
                trace = [(tt - t0 + offset, q_, jac, valid) for tt, q_, jac, valid in phi.trace]
                top_jac = [np.count_nonzero(m & planted) / np.count_nonzero(m | planted) for _, _, m in top]
                ranks = [r + 1 for r, jac in enumerate(top_jac) if jac >= HIT_JACCARD]
                row = dict(spec)
                row.update({"run": RUN, "method": name, "depth_searched": d, "n_attrs": A_.shape[1],
                            "conditions": int(sum(len(f) for f in freqs)), "coverage": planted.mean(),
                            "planted_n": int(planted.sum()),
                            "evaluated_hit": any(valid and jac >= HIT_JACCARD for _, _, jac, valid in trace),
                            "rank": ranks[0] if ranks else np.nan, "evaluated": len(trace),
                            "runtime_s": time.time() - t0 + offset, "seed_time_s": offset, **info})
                for bgt in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                    row[f"rank_at_{bgt}"] = _rank_of_planted(trace[:bgt])
                for sec in [1, 2, 5, 10, 30, 60]:
                    row[f"rank_at_{sec}s"] = _rank_of_planted([e for e in trace if e[0] <= sec])
                out.append(row)
                continue
            cands, spaces, truncated = bart_candidates(pairs, A_, d, CAND_BUDGET)
            offset = dtime + time.time() - t
            pset = set(planted_desc)
            info = {"mini_clusters": len(pairs), "mean_mc_size": 2.0,
                    "best_mc_purity": max((planted[list(pr)].mean() for pr in pairs), default=0.0),
                    "planted_conds_in_basket": max((len(set(x) & pset) for x in spaces), default=0),
                    "planted_in_space": any(pset <= set(x) for x in spaces),
                    "shared_per_pair": float(np.mean([len(x) for x in spaces])) if spaces else 0.0,
                    "candidates": len(cands), "budget_hit": truncated,
                    "pair_level_pct": float(np.mean([LEVEL_PCT[list(pr)].mean() for pr in pairs])) if pairs else np.nan}
            t0 = time.time()
            top, stats = evaluate_candidates(A_, phi, cands, prune=True)
            # optimistic bound for an uncapped run: if the true description is in Bart's candidate space, would it
            # enter the top-q next to what was evaluated? (more candidates could only push it down)
            if info["planted_in_space"]:
                phi_free = MonetaryQuality(exc, level, store_codes, relative=True, min_cov=MIN_COV_V5)
                pmask = mask_of(A_, planted_desc)
                q_true = phi_free(pmask)
                info["found_unlimited_bound"] = bool(phi_free.satisfies_constraints(pmask) and
                                                     (len(top) < Q or q_true >= top[Q - 1][0]))
            else:
                info["found_unlimited_bound"] = False
        else:
            b, offset, info = baskets(s)
            if s["kind"] == "pool":
                votes = Counter(c for x in b for c in x)
                top, stats = beam_search(A_, phi, w=10, d=d, prune=True, pool=[c for c, _ in votes.most_common(POOL_SIZE)])
            elif s["kind"] == "itemsets":
                support = itemset_support(b, d)
                cands = {c for c, n in support.items() if n >= s["minsup"]}
                info = dict(info, itemset_candidates=len(cands), planted_support=support.get(planted_desc, 0))
                top, stats = evaluate_candidates(A_, phi, cands, prune=True)
                if s["guided"]:
                    rs = ResultSet(Q)
                    for q_, d_, m_ in top:
                        rs.insert(q_, d_, m_)
                    top = guided_beam(A_, phi, top, rs, stats, d, support)
            else:
                cands = set()
                for x in b:
                    for size in range(1, min(d, len(x)) + 1):
                        cands.update(tuple(sorted(c)) for c in itertools.combinations(x, size))
                info = dict(info, candidates=len(cands))
                top, stats = evaluate_candidates(A_, phi, cands, prune=True)
                if s.get("local"):
                    rs = ResultSet(Q)
                    for q_, d_, m_ in top:
                        rs.insert(q_, d_, m_)
                    top = local_search_d(A_, phi, top, rs, stats, d)
        trace = [(tt - t0 + offset, q_, jac, valid) for tt, q_, jac, valid in phi.trace]
        top_jac = [np.count_nonzero(m & planted) / np.count_nonzero(m | planted) for _, _, m in top]
        ranks = [r + 1 for r, jac in enumerate(top_jac) if jac >= HIT_JACCARD]
        row = dict(spec)
        row.update({"run": RUN, "method": name, "depth_searched": d, "n_attrs": A_.shape[1],
                    "conditions": int(sum(len(f) for f in freqs)), "coverage": planted.mean(),
                    "planted_n": int(planted.sum()),
                    "evaluated_hit": any(valid and jac >= HIT_JACCARD for _, _, jac, valid in trace),
                    "rank": ranks[0] if ranks else np.nan, "evaluated": len(trace),
                    "runtime_s": time.time() - t0 + offset, "seed_time_s": offset,
                    "local_evaluated": stats.get("local_evaluated", 0), **info})
        for bgt in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
            row[f"rank_at_{bgt}"] = _rank_of_planted(trace[:bgt])
        for sec in [1, 2, 5, 10, 30, 60]:
            row[f"rank_at_{sec}s"] = _rank_of_planted([e for e in trace if e[0] <= sec])
        out.append(row)
    return out
'''

SUITE_RUN = r'''
# ---- calibration (deterministic): null spread of random true-subgroup-like descriptions, per depth
rng = np.random.default_rng(SEED + 7)
SIGMA_LITS = {}
for lits in LITS_V5:
    rates = []
    while len(rates) < 200:
        A_ = make_descriptors(rng, 200)
        desc, mask = sample_description_cat(rng, A_, value_freqs(A_), lits, store_codes)
        if desc is not None:
            rates.append(excess[mask][:, POST].sum() / (level[mask].sum() * POST.sum()))
    SIGMA_LITS[lits] = float(np.std(rates))
print("null spread per description length:", {k: round(v, 4) for k, v in SIGMA_LITS.items()})

D_blocks_by = {b: pairwise_cityblock(block_means_n(rel_dev, b)) for b in BLOCK_SIZES}
# full-T distances (dollar excess / relative deviation) are only needed for the seed ablation
D_money_full = pairwise_cityblock(excess) if NEEDS_FULL_DISTANCES else None
D_shape_full = pairwise_cityblock(rel_dev) if NEEDS_FULL_DISTANCES else None
# Bart's original BUS: squared Euclidean (eq. 2.15) on the full target, as-is ($ excess) and scale-free (relative deviation)
D_orig_money = pairwise_sqeuclidean(excess) if NEEDS_ORIGINAL else None
D_orig_rel = pairwise_sqeuclidean(rel_dev) if NEEDS_ORIGINAL else None

specs = []
for k in K_ATTRS:
    for lits in LITS_V5:
        for snr, outliers in GRID:
            for rep in range(REPS_V5):
                specs.append({"dataset": DATASET, "k": k, "lits": lits, "snr": snr, "outliers": outliers, "rep": rep,
                              "delta": snr * SIGMA_LITS[lits], "seed": int(rng.integers(1 << 31))})

out_csv = f"{RESULTS_DIR}/tasks.csv"
progress = f"{RESULTS_DIR}/progress.txt"
done = set(pd.read_csv(out_csv)["seed"]) if os.path.exists(out_csv) else set()
skipped_file = f"{RESULTS_DIR}/skipped_seeds.txt"
skipped = set(int(x) for x in open(skipped_file).read().split()) if os.path.exists(skipped_file) else set()
todo = sorted([s for s in specs if s["seed"] not in done and s["seed"] not in skipped],
              key=lambda s: (s["rep"], s["lits"], s["k"]))
print(f"{RUN}: {len(specs)} tasks x {len(SUITE_METHODS)} methods; {len(done) + len(skipped)} already done, {len(todo)} to run")
start = time.time()
for b in range(0, len(todo), BATCH):
    batch = todo[b:b + BATCH]
    parts = Parallel(n_jobs=N_JOBS)(delayed(_suite_job)(s, excess, level, store_codes, D_blocks_by, D_money_full,
                                                        D_shape_full, METHOD_SPECS, D_orig_money, D_orig_rel)
                                    for s in batch)
    rows = [r for p in parts for r in p]
    if rows:
        new = pd.DataFrame(rows)
        if os.path.exists(out_csv):            # same columns, same order as the file (a missing value stays empty)
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
        f.write(f"{RUN}: {len(done) + len(skipped) + b + len(batch)} / {len(specs)} tasks, {elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left\n")
print(f"done in {time.time() - start:.0f}s")
'''

SUITE_SUMMARY = r'''
from scipy.stats import binomtest

res = pd.read_csv(f"{RESULTS_DIR}/tasks.csv")
res["found"] = res["rank"] <= Q
REF = "BS w=10"
key = ["k", "lits", "snr", "outliers", "rep"]
n_tasks = res.drop_duplicates(key).shape[0]
print(f"{RUN}: {n_tasks} tasks built (of {len(specs)}; the rest had no valid true subgroup)")
summary = (res.groupby("method").agg(found=("found", "mean"), generated=("evaluated_hit", "mean"),
                                     evaluated=("evaluated", "mean"), runtime_s=("runtime_s", "mean"))
           .reindex(SUITE_METHODS))
display(summary.round(3))
display(res.pivot_table(index=["k", "lits"], columns="method", values="found")[SUITE_METHODS].round(2))
w = res.pivot_table(index=key, columns="method", values="found").astype(bool)
groups = [("all", w)] + [(f"lits={l}", w.xs(l, level="lits", drop_level=False)) for l in sorted(res.lits.unique())] \
       + [(f"k={k}", w.xs(k, level="k", drop_level=False)) for k in sorted(res.k.unique())] \
       + [(f"snr={s}, outliers={o}", w.xs((s, o), level=("snr", "outliers"), drop_level=False))
          for s, o in sorted(set(zip(res.snr, res.outliers)))]
rows = []
for g, sub in groups:
    for a in SUITE_METHODS:
        if a == REF:
            continue
        wins, losses = int((sub[a] & ~sub[REF]).sum()), int((~sub[a] & sub[REF]).sum())
        rows.append({"run": RUN, "group": g, "method": a, "found": sub[a].mean(), "found_ref": sub[REF].mean(),
                     "wins": wins, "losses": losses, "p": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0,
                     "n": len(sub)})
paired = pd.DataFrame(rows)
paired.to_csv(f"{RESULTS_DIR}/paired_vs_BS10.csv", index=False)
if "BUS" in SUITE_METHODS:          # the same McNemar test against fixed BUS
    rows = []
    for g, sub in groups:
        for a in SUITE_METHODS:
            if a == "BUS":
                continue
            wins, losses = int((sub["BUS"] & ~sub[a]).sum()), int((~sub["BUS"] & sub[a]).sum())
            rows.append({"run": RUN, "group": g, "method": a, "found_BUS": sub["BUS"].mean(), "found": sub[a].mean(),
                         "BUS_wins": wins, "BUS_losses": losses,
                         "p": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0, "n": len(sub)})
    paired_bus = pd.DataFrame(rows)
    paired_bus.to_csv(f"{RESULTS_DIR}/paired_vs_BUS.csv", index=False)
    display(paired_bus[paired_bus.group == "all"].round(4))
summary.assign(run=RUN, tasks=n_tasks).to_csv(f"{RESULTS_DIR}/summary.csv")
display(paired[paired.group == "all"].round(4))
'''


def load_nb(path):
    return nbformat.read(path, as_version=4)


template = load_nb("m5_cat_v7.ipynb")
loaders = {"walmart": load_nb("walmart_subtle.ipynb"), "favorita": load_nb("favorita_subtle.ipynb")}

for run, cfg in RUNS.items():
    if ONLY and run not in ONLY:
        continue
    nb = load_nb("m5_cat_v7.ipynb")
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    if cfg["target"] != "m5":
        src_nb = loaders[cfg["target"]]
        nb.cells[2].source = src_nb.cells[2].source        # dataset settings
        nb.cells[6].source = src_nb.cells[3].source        # data loading
    nb.cells[2].source = nb.cells[2].source.replace("QUICK = False", f"QUICK = {QUICK}")
    assert f"QUICK = {QUICK}" in nb.cells[2].source
    grid = cfg.get("grid", [(8, 0.0), (8, 0.2), (16, 0.0), (16, 0.2)])
    nb.cells[0].source = f"# Weekend suite · {run}\n{cfg['why']}\n\nCategorical semi-synthetic design as v7 (see m5_cat_v7.ipynb); target = {cfg['target']}; " \
                         f"generator = {cfg['generator']}; depth = {cfg['depth']}; methods: {', '.join(cfg['methods'])}.\n" \
                         "Resumable: progress in results_suite/<run>/progress.txt; rerunning skips finished tasks."
    settings = SUITE_SETTINGS.format(run=run, generator=cfg["generator"], grid=grid, reps=cfg["reps"], depth=cfg["depth"],
                                     methods=cfg["methods"], needs=cfg.get("needs_full_distances", False),
                                     needs_orig=cfg.get("needs_original", False), why=cfg["why"])
    new_cells = nb.cells[:14] + [nbformat.v4.new_code_cell(settings.strip()), nbformat.v4.new_code_cell(SUITE_CORE.strip()),
                                 nbformat.v4.new_code_cell(SUITE_RUN.strip()), nbformat.v4.new_code_cell(SUITE_SUMMARY.strip())]
    nb.cells = new_cells
    nbformat.write(nb, f"suite_{run}.ipynb")
    print("wrote", f"suite_{run}.ipynb")
