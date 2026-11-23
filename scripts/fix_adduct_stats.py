"""Recompute per-adduct observed-offset and window-tolerance stats for mass matching.

Definition (verified against train):
    ppm = (precursor_mz - neutral_mass)/neutral_mass * 1e6
offset[a] = median(ppm) over train rows with adduct a
tol[a]    = quantile(|ppm - offset[a]|, 0.999) + 1.0 (capped at 12 ppm)
mass_candidates accepts candidates with |ppm - offset[a]| <= tol[a], i.e. the
acceptance window on the neutral mass is centered on the observed offset.
"""
import json

import numpy as np
import pandas as pd

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
pool = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/pool.parquet")
key_col = next(c for c in tr.columns if "key14" in c)

mass = pool["mass"]
sub = tr[tr.adduct.isin([c for c in tr.adduct.unique() if isinstance(c, str)])].copy()
sub["mass"] = sub[key_col].map(mass)
sub = sub.dropna(subset=["mass"])
sub["ppm"] = (sub.precursor_mz - sub.mass) / sub.mass * 1e6

stats = {}
rows = []
for a in sorted(sub.adduct.unique()):
    e = sub.loc[sub.adduct == a, "ppm"]
    if len(e) == 0:
        continue
    off = float(np.median(e))
    dev = np.abs(e - off)
    tol = min(float(np.quantile(dev, 0.999)) + 1.0, 12.0)
    stats[str(a)] = {"offset_ppm": off, "tol_ppm": tol, "n": int(len(e))}
    rows.append((a, len(e), off, tol))

print(pd.DataFrame(rows, columns=["adduct", "n", "offset_ppm", "tol_ppm"]).to_string(index=False))
stats["__default__"] = {
    "offset_ppm": float(np.median(sub["ppm"])),
    "tol_ppm": min(float(np.quantile(np.abs(sub["ppm"] - np.median(sub["ppm"])), 0.999)) + 1.0, 12.0),
    "n": int(len(sub)),
}
with open("d:/Dev/KaggleEnvedaCASMI/data/adduct_offsets.json", "w") as fh:
    json.dump(stats, fh, indent=1)
print("wrote data/adduct_offsets.json")
