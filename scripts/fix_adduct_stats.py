"""Recompute per-adduct shifts + tolerances for mass matching (Da-based).

Semantics (verified against train):
  d_da(row) = precursor_mz - neutral_mass (Da, per train row)
  shift[a]  = median of d_da over all rows of adduct a (systematic shift, Da)
  per molecule: q_dev = median of its rows' (d_da - shift[adduct])
  tol[a]    = quantile(|q_dev| over molecules, 0.999) + 0.5
mass_candidates accepts candidates with |prec - m - shift[a]| <= tol[a]: the
neutral-mass window is [prec - shift - tol, prec - shift + tol].
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
pool = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/pool.parquet")
key_col = next(c for c in tr.columns if "key14" in c)

tr["mass"] = tr[key_col].map(pool["mass"])
tr["d_da"] = tr["precursor_mz"] - tr["mass"]
shift = tr.groupby("adduct")["d_da"].median().to_dict()


def mol_dev(d):
    return float((d["d_da"] - d["adduct"].map(shift)).median())


groups = {}
for key, d in tr.groupby(key_col):
    med = d["precursor_mz"].median()
    i = d["precursor_mz"].sub(med).abs().idxmin()
    groups[key] = (str(d.at[i, "adduct"]), mol_dev(d))

by_ad = defaultdict(list)
for k, (a, qd) in groups.items():
    by_ad[a].append((k, qd))

stats = {}
for a, lst in by_ad.items():
    s = float(shift.get(a, 0.0))
    df = pd.DataFrame(lst, columns=["key", "q_dev"])
    dev = df["q_dev"].abs()
    tol = min(float(np.quantile(dev, 0.999)) + 0.5, 3.0)
    frac = float((dev <= tol).mean())
    if a in ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"):
        print(f"{a}: shift={s:.4f}Da tol={tol:.4f}Da frac<=tol={frac:.3f} n={len(df)}")
    stats[a] = {"shift_da": s, "tol_da": tol, "n": int(len(df))}

stats["__default__"] = {"shift_da": float(np.median(tr["d_da"])), "tol_da": 1.0, "n": int(len(tr))}
with open("d:/Dev/KaggleEnvedaCASMI/data/adduct_offsets.json", "w") as fh:
    json.dump(stats, fh, indent=1)
print("wrote data/adduct_offsets.json")
