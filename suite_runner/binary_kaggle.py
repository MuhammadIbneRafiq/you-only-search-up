"""Binary study on Kaggle (or anywhere): the same code as the local run (make_binary.py + m5_deep_v5.ipynb cells),
plus the TODO methods, with a time budget and resume.

    from binary_kaggle import run
    run("favorita")                 # or "walmart" / "m5"

- Tasks are regenerated from the same seeds as the local run, so results pair task-by-task with results_binary/.
- Stops starting new batches before `hours` run out, so a Kaggle "Save & Run All" (12 h cap) ends cleanly.
- Resume: attach the previous version's output as an input; its results_todo/<dataset>/ files are picked up and
  only the remaining tasks are run.
"""
import glob
import json
import os
import shutil
import time
from pathlib import Path

import make_binary as mb

REPO = Path(__file__).resolve().parent.parent

# BS w=10, BUS snn and the pool are references measured on the same machine; "+local" continues from the method
# right before it. New: local traversal after beam search, at every level of BUS snn, and after the SNN pool.
TODO_METHODS = ["BS w=10", "BS w=10 +local",
                "BUS snn", "BUS snn +local", "BUS snn +local per level",
                "BUS→BS pool snn", "BUS→BS pool snn +local"]

# the new method kinds, spliced into the unchanged core code
_PARSE_OLD = '''    if name.startswith("BS"):
        return {"kind": "bs", "w": int(name.split("w=")[1]), "local": False}'''
_PARSE_NEW = '''    if name.startswith("BS"):
        local = name.endswith("+local")
        return {"kind": "bs", "w": int(name.split("w=")[1].split()[0]), "local": local,
                "base": name.replace(" +local", "") if local else None}
    if name.endswith("+local per level"):
        return {"kind": "perlevel", "seeds": "snn", "local": False, "base": None}'''
_BRANCH_OLD = '''            elif s["seeds"] == "snn":
                cands = set()'''
_BRANCH_NEW = '''            elif s["kind"] == "perlevel":
                top, lstats = bus_levels_local(A_, phi, baskets, d)
                info["local_evaluated"] = lstats["local_evaluated"]
            elif s["seeds"] == "snn":
                cands = set()'''
_LEVELS = r'''
def bus_levels_local(A_, phi, baskets, d):
    """BUS snn with local traversal at every level: evaluate BUS's candidates with 1 condition, then one round of local
    moves (drop / add / swap one condition, at most `size` conditions) from the current 10 best; then the candidates
    with 2 conditions and another round; ... up to depth d. Same lossless pruning as evaluate_candidates."""
    rs, stats, dead = ResultSet(Q), Counter(), set()
    for size in range(1, d + 1):
        cands = set()
        for b in baskets:
            if len(b) >= size:
                cands.update(tuple(sorted(c)) for c in itertools.combinations(b, size))
        for desc in sorted(cands):
            if size > 1 and any(sub in dead for k in range(1, size) for sub in itertools.combinations(desc, k)):
                continue
            mask = mask_of(A_, desc)
            if not phi.expandable(mask):
                dead.add(desc)
                continue
            quality = phi(mask)
            if phi.satisfies_constraints(mask):
                rs.insert(quality, desc, mask)
            if oe_below_threshold(phi, mask, rs):
                dead.add(desc)
        local_search_d(A_, phi, rs.sorted(), rs, stats, size, max_rounds=1)
    return rs.sorted(), stats
'''
_DBART_OLD = "D_bart = pairwise_sqeuclidean(excess)                      # Bart's BUS: squared Euclidean on the target as-is"
_DBART_NEW = ("D_bart = pairwise_sqeuclidean(excess) if any(s.get('seeds') in ('closest', 'alpha') "
              "for s in METHOD_SPECS.values()) else None   # only Bart's BUS needs it")
_LOOP_OLD = '''for b in range(0, len(todo), BATCH):
    batch = todo[b:b + BATCH]'''
_LOOP_NEW = '''last_batch_s = 0.0
for b in range(0, len(todo), BATCH):
    if time.time() + 1.5 * last_batch_s > DEADLINE:
        print(f"time budget reached: stopping after {b} of {len(todo)} remaining tasks; rerun to continue")
        break
    batch_start = time.time()
    batch = todo[b:b + BATCH]'''
_LOOP_END_OLD = '''print(f"done in {time.time() - start:.0f}s")'''
_LOOP_END_NEW = '''    last_batch_s = time.time() - batch_start
print(f"done in {time.time() - start:.0f}s")'''


def _splice(text, old, new):
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def _cells(path):
    return ["".join(c["source"]) for c in json.load(open(path, encoding="utf-8"))["cells"]]


def _find(name, base):
    hits = [p for p in glob.glob(f"{base}/**/{name}", recursive=True) if "/results" not in p.replace("\\", "/")]
    return Path(hits[0]).resolve() if hits else None


def _merge_previous(out_dir, dataset, results_name, base):
    """Collect earlier results of this dataset (attached Kaggle outputs under `base`, or a previous session in
    out_dir) into out_dir: tasks.csv deduplicated on (seed, method), skipped seeds unioned."""
    import pandas as pd
    target = Path(out_dir) / dataset
    target.mkdir(parents=True, exist_ok=True)
    found = [Path(p) for p in glob.glob(f"{base}/**/{results_name}/{dataset}/tasks.csv", recursive=True)]
    if (target / "tasks.csv").exists():
        found.append(target / "tasks.csv")
    if found:
        frames = [pd.read_csv(p) for p in dict.fromkeys(found)]
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(["seed", "method"])
        merged = merged[merged.groupby("seed").method.transform("size") == merged.method.nunique()]   # whole tasks only
        merged.to_csv(target / "tasks.csv", index=False)
    skipped = set()
    for p in glob.glob(f"{base}/**/{results_name}/{dataset}/skipped_seeds.txt", recursive=True) + \
             [str(target / "skipped_seeds.txt")]:
        if os.path.exists(p):
            skipped |= set(open(p).read().split())
    if skipped:
        open(target / "skipped_seeds.txt", "w").write("\n".join(sorted(skipped)) + "\n")
    n = 0 if not (target / "tasks.csv").exists() else pd.read_csv(target / "tasks.csv").seed.nunique()
    print(f"resume: {n} finished tasks and {len(skipped)} skipped seeds carried over")


def run(dataset, hours=11.0, methods=None, data_base="/kaggle/input", out_dir="/kaggle/working/results_todo",
        n_jobs=None, dry_run=False):
    """Run (or continue) the binary study for one dataset. Returns the namespace (results in ns['res'])."""
    deadline = time.time() + hours * 3600
    methods = list(methods or TODO_METHODS)
    out_dir = str(Path(out_dir).resolve())
    data_root = _find("final_sales_dataset.csv", data_base)
    assert data_root is not None, f"final_sales_dataset.csv not found under {data_base}"
    data_root = data_root.parent
    print("data:", data_root)
    _merge_previous(out_dir, dataset, Path(out_dir).name, data_base)

    template = _cells(REPO / "m5_deep_v5.ipynb")
    if dataset != "m5":
        loader = _cells(REPO / f"{dataset}_subtle.ipynb")
        template[2], template[6] = loader[2], loader[3]
    scratch = Path(out_dir) / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    from IPython.display import display
    ns = {"__name__": "binary_study", "display": display}
    cwd = os.getcwd()
    os.chdir(scratch)                         # the template cells create a few result folders; keep them out of the way
    try:
        for i in range(1, 14):
            src = template[i]
            if i == 2:
                assert "QUICK = False" in src
            if i == 6:
                os.chdir(data_root)           # Walmart / Favorita loaders read relative paths
            exec(compile(src, f"<template cell {i}>", "exec"), ns)
            if i == 6:
                os.chdir(scratch)
            if i == 2:
                ns["M5_DIR"] = str(data_root / "M5")
        ns["QUICK"] = False
        exec(mb.SETTINGS.format(methods=methods), ns)
        ns["RESULTS_DIR"] = f"{out_dir}/{dataset}"
        os.makedirs(ns["RESULTS_DIR"], exist_ok=True)
        if n_jobs:
            ns["N_JOBS"], ns["BATCH"] = n_jobs, 2 * n_jobs
        core = _splice(_splice(mb.CORE, _PARSE_OLD, _PARSE_NEW), _BRANCH_OLD, _BRANCH_NEW) + _LEVELS
        exec(core, ns)
        run_code = _splice(mb.RUN, _DBART_OLD, _DBART_NEW)
        specs_part, loop_part = run_code.split("out_csv, progress =", 1)
        exec(specs_part, ns)
        if dry_run:
            return ns
        ns["DEADLINE"] = deadline
        loop_part = _splice(_splice("out_csv, progress =" + loop_part, _LOOP_OLD, _LOOP_NEW), _LOOP_END_OLD, _LOOP_END_NEW)
        exec(loop_part, ns)
        exec(mb.SUMMARY, ns)
    finally:
        os.chdir(cwd)
    shutil.rmtree(scratch, ignore_errors=True)
    return ns
