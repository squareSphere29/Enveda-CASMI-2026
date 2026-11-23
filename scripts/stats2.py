import pandas as pd
import numpy as np

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")

KEY_ADDUCTS = [
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-",
    "[2M+H]+", "[2M+Na]+", "[2M+H-O2]2-", "[M+HNa]2+", "[M+Ca]2+",
    "[M+2H]2+", "[M+2Na-H]+", "[2M+Na-H2O]+", "[M+CH2O2-H]-", "[M+M+H]+",
]
sub = tr[tr.adduct.isin(KEY_ADDUCTS)].copy()
print("rows by key adducts:", len(sub), "of", len(tr))
q = sub.groupby("adduct").precursor_error_ppm.agg(
    n="size", med="median", p05=lambda s: s.quantile(0.05), p50=lambda s: s.quantile(0.5),
    p95=lambda s: s.quantile(0.95), p99=lambda s: s.quantile(0.99),
    p999=lambda s: s.quantile(0.999), mx="max", mn="min").round(4)
print(q)
print()
# dup check: how many inchikey14 values have >1 distinct smiles
dups = tr.groupby("inchikey14")["normalized_smiles"].nunique()
print("keys with >1 distinct smiles:", (dups > 1).sum(), "of", len(dups))
print("smiles-per-key distribution:", dups.value_counts().head(8).to_dict())
print()
# per molecule: number of distinct spectra (rows)
mols = tr.groupby("inchikey14").agg(
    rows=("ms2_mzs", "size"), smiles=("normalized_smiles", "nunique"),
    adducts=("adduct", lambda s: s.nunique()), lib=("ingest_lib", "first"))
print("spectra rows per molecule: median", mols.rows.median(), "p95", mols.rows.quantile(0.95), "max", mols.rows.max())
print("molecules with 0 spectra:", (mols.rows == 0).sum())
print()
# np-examples panel
npv = tr[tr.ingest_lib == "enveda-np-examples"]
print("enveda-np-examples rows:", len(npv), "unique molecules:", npv.inchikey14.nunique())
print("np molecules in train overall:", tr[tr.inchikey14.isin(npv.inchikey14.unique())].inchikey14.nunique())
print()
print("test adducts:")
te = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/test.parquet")
print(te.adduct.value_counts().head(20).to_dict())
print("ionization:", te.ionization_mode.value_counts().to_dict())
print("instrument:", te.instrument_type.value_counts().dropna().to_dict())
print("test rows:", len(te), "unique molecule_id:", te.molecule_id.nunique())
print("precursor range:", te.precursor_mz.min().round(4), te.precursor_mz.max().round(4))
print("num peaks: median", te.num_peaks.median() if "num_peaks" in te else np.median([len(x) for x in te.ms2_mzs]))
print("collision energy orig sample:", te.collision_energy_orig.dropna().unique()[:8])
