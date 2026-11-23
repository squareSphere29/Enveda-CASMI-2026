"""Inspect precursor vs neutral mass per adduct: medians + quantiles for window design."""
import numpy as np
import pandas as pd

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
pool = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/pool.parquet")
key_col = next(c for c in tr.columns if "key14" in c)
print("key column:", repr(key_col))

mass = pool["mass"]
sub = tr[tr.adduct.isin(["[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-",
                         "[2M+H]+", "[M+2H]2+", "[M+HNa]2+"])].copy()
sub["mass"] = sub[key_col].map(mass)
sub = sub.dropna(subset=["mass"])
sub["ppm_vs_neutral"] = (sub.precursor_mz - sub.mass) / sub.mass * 1e6
sub["col_err_ppm"] = sub.precursor_error_ppm

g = sub.groupby("adduct")
tab = pd.DataFrame({
    "n": g.size(),
    "med_ppm_vs_neutral": g.ppm_vs_neutral.median(),
    "med_col_err": g.col_err_ppm.median(),
    "p999_abs_dev_ppm_vs_neutral": g.apply(
        lambda d: np.quantile(np.abs(d.ppm_vs_neutral - np.median(d.ppm_vs_neutral)), 0.999),
        include_groups=False),
})
print(tab.round(3).to_string())
print()
print("sample rows [M+H]+:")
print(sub[sub.adduct == "[M+H]+"][["precursor_mz", "mass", "ppm_vs_neutral", "col_err_ppm"]].head(6).to_string(index=False))
