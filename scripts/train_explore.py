import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

pd.set_option("display.width", 200)

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
print("train rows:", len(tr))
print("unique inchikey14:", tr.inchikey14.nunique())
print("unique normalized_smiles:", tr.normalized_smiles.nunique())
print("ingest_lib:", tr.ingest_lib.value_counts().to_dict())
print("ionization_mode:", tr.ionization_mode.value_counts().to_dict())
print("adduct:", tr.adduct.value_counts().to_dict())
print("instrument_type:", tr.instrument_type.value_counts().to_dict())
print("ppm: describe", tr.precursor_error_ppm.describe().round(4).to_dict())
print()
ppm = tr.precursor_error_ppm.dropna()
print(ppm.groupby(tr.ionization_mode).agg(["median", "mean", "std", "min", "max", lambda s: s.quantile(0.01), lambda s: s.quantile(0.99)]).round(4))
print()
ppm2 = tr.assign(_i=tr.ionization_mode, _a=tr.adduct).groupby(["_i", "_a"]).precursor_error_ppm.agg(["median", "mean", "count"]).round(4)
print(ppm2)
print()
print(tr.groupby(["ingest_lib", "ionization_mode"]).size())
print()
# sample rows
for i in range(3):
    r = tr.iloc[i]
    print({k: (v[:80] if isinstance(v, str) and len(v) > 80 else v) for k, v in r.items() if k not in ("ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev")})
    print("  ms2 peaks:", len(r.ms2_mzs))

# verify InChIKey14 computation from smiles on a sample
taut = rdMolStandardize.TautomerEnumerator()
ok = bad = 0
bad_examples = []
sm2ik = tr.drop_duplicates("normalized_smiles").set_index("normalized_smiles")["inchikey14"].to_dict()
for smi in list(tr.normalized_smiles.unique())[:4000]:
    ik14 = sm2ik.get(smi)
    if ik14 is None:
        continue
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    mc = taut.Canonicalize(m)
    ik = inchi.MolToInchiKey(mc)
    if ik[:14] == ik14:
        ok += 1
    else:
        bad += 1
        if len(bad_examples) < 5:
            bad_examples.append((smi[:40], ik14, ik[:14]))
print("InChIKey14 computed==column: ok", ok, "bad", bad)
for b in bad_examples:
    print("  mismatch example:", b)
