"""Per-adduct Da shift and a tight tolerance from train rows of that adduct only.

Mixing adducts inside a molecule made the old 0.999 quantile + 0.5 Da hit the 3 Da
cap, which retrieved thousands of candidates per query.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

tr = pd.read_parquet(DATA / "train.parquet", columns=["inchikey14", "precursor_mz", "adduct"])
pool = pd.read_parquet(DATA / "pool.parquet")
mass = pool["mass"]
tr = tr.join(mass, on="inchikey14")
tr = tr[np.isfinite(tr["mass"]) & np.isfinite(tr["precursor_mz"])]
tr["d_da"] = tr["precursor_mz"] - tr["mass"]

stats = {}
for a, d in tr.groupby("adduct"):
    a = str(a)
    shift = float(d["d_da"].median())
    err = (d["d_da"] - shift).to_numpy()
    q = float(np.quantile(np.abs(err), 0.995))
    tol = float(min(max(q, 0.005), 0.05))
    stats[a] = {"shift_da": shift, "tol_da": tol, "n": int(len(d))}
    if a in ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-",
             "[2M+H]+", "[2M+Na]+"):
        print(f"{a}: shift={shift:.4f}Da tol={tol:.4f}Da n={len(d)} q995={q:.4f}")

all_err = tr["d_da"] - tr["adduct"].map({a: stats[a]["shift_da"] for a in stats})
stats["__default__"] = {
    "shift_da": float(tr["d_da"].median()),
    "tol_da": float(min(max(np.quantile(np.abs(all_err), 0.995), 0.005), 0.05)),
    "n": int(len(tr)),
}
out = DATA / "adduct_offsets.json"
out.write_text(json.dumps(stats, indent=1))
print("wrote", out)
