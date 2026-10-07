import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import itertools


class RankBiserial:
    """
    Compares between-distance and within-distance with within meaning both series are in subgroup s,
    and between meaning one series is in s and the other is not. The rank-biserial correlation
    is the difference between the probability that a between-distance is greater than a within-distance
    r = P(between > within) - P(between < within), which is equivalent to r = 2 * U / (nm) - 1, where
    U is the number of combinations where between distance > within distance.
    """

    def __init__(self, Y, MIN_COV):
            self.D = cdist(Y, Y)
            self.rowsum = self.D.sum(1)
            self.MIN_COV = MIN_COV
            self.cache = {}
    
    def __call__(self, m):
        key = m.tobytes()
        if key not in self.cache:
            n = m.sum()
            if n < max(2, self.MIN_COV * len(m)) or n == len(m):
                self.cache[key] = None
            else:
                s = self.D @ m
                within = s[m].sum()
                beta = within / (n * (n - 1))
                alpha = (self.rowsum[m].sum() - within) / (n * (len(m) - n))
                self.cache[key] = alpha / (beta + 1)
        return self.cache[key]
    
def mask_of(A, desc):
    mask = np.ones(len(A), dtype=bool)

    for column_index, value in desc:
        mask &= (A[:, column_index] == value)

    return mask

def all_conditions(A):
    # Creates conditions such as Type = A, Store = 1, CPI_mean_category = high
    return [
        (column_index, value)
        for column_index in range(A.shape[1])
        for value in pd.unique(A[:, column_index])
    ]

# Define BUSPaS, which generates subgroup candidates from pairs of
# similar sales trajectories and their shared descriptor values.
def describe(desc, names):
    return ' AND '.join(names[j] if v == 1 else f'NOT {names[j]}' for j, v in desc)


# SEARCHES
def beam_search(A, qual, K, w, D, positive_only=True):
    # D = max depth
    # w = beam width
    # A = binary matrix of descriptors (rows = series, cols = attribute==value)
    # qual = quality measure (callable)
    # K = best K descriptors to return
    
    k = A.shape[1]
    values = (1,) if positive_only else (0, 1)

    # 'conds' has that (j, v) is a condition with column j of A and value v
    conds = [(j, v) for j in range(k) for v in values]
    
    # 'seeds' starts from empty descriptor, then grows by adding one condition at a time
    results, seeds, seen = {}, [()], set()
    for _ in range(D):
        beam = []
        for seed in seeds:
            used = {j for j, _ in seed}
            for c in conds:
                if c[0] in used:
                    continue
                # add condition c to the current seed descriptor, sort to canonical order
                desc = tuple(sorted(seed + (c,)))
                if desc in seen:
                    continue
                seen.add(desc)
                m = mask_of(A, desc)
                phi = qual(m)
                if phi is not None:
                    results[desc] = (phi, m)
                    beam.append((phi, desc))
        # keep top w descriptors w/ highest quality
        beam.sort(reverse=True)
        seeds = [d for _, d in beam[:w]]
    best = {}
    for desc, (phi, m) in results.items():
        key = m.tobytes()
        if key not in best or len(desc) < len(best[key][0]):
            best[key] = (desc, phi, m)
    # return top K descriptors w/ highest quality across all levels
    top = sorted(best.values(), key=lambda x: -x[1])[:K]
    return top, len(seen)


def bottom_up_search(A, qual, D):   
    # The distance measure in the quality measure given
    D = qual.D

    iu = np.triu_indices(len(D), 1)
    flat = D[iu]
    best = np.argpartition(flat, U)[:U]
    pairs = zip(iu[0][best], iu[1][best])
    cands = set()

    results = {}
    seeds = [()]
    seen = set()

    for i, j in pairs:
        shared = [(a, A[i, a]) for a in range(A.shape[1]) if A[i, a] == A[j, a]]

        for d in range(1, D + 1):
            #if desc in seen:
            #    continue
            #seen.add(desc)
            cands.update(itertools.combinations(shared, d))
            # m = mask_of(A, desc)
            # phi = qual(m)
            # if phi is not None:
            #     results[desc] = (phi, m)

    # Now we make a list of all our subgroups
    results = {}
    for desc in cands:
        m = mask_of(A, desc)
        key = m.tobytes()
        phi = qual(m)
        if phi is not None:
            results[key] = (desc, phi, m)
    top = sorted(results.values(), key=lambda x: -x[1])[:K]
    return top, len(cands)