"""Build the molecule library for the CASMI26 solution.

- Pool: one row per unique inchikey14 (the metric key from train.parquet).
- Per molecule: normalized SMILES, formula, mass (exact, from SMILES),
  ingest_lib, doc_prior (natural-product / documentation prior), NP features,
  and the per-molecule concatenated library spectra (mz, normalized intensity).
- Per-adduct: observed offset (ppm vs neutral mass) and window tolerance,
  calibrated from train.parquet.

Outputs (data/): pool.parquet, pool_spec.npz, adduct_offsets.json
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

DATA = str(Path(__file__).resolve().parent.parent / "data")
FP_SIZE = 2048
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


def parse_formula(f):
    return {e: int(n) if n else 1 for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f or "") if e}


def np_features(m, formula):
    d = {"n_rings": 0, "n_arom_atoms": 0, "n_N": 0, "n_O": 0, "n_S": 0,
         "n_hal": 0, "n_hetero": 0, "n_heavy": 0, "n_double_bonds": 0,
         "n_rot": 0, "DBE": 0.0}
    if m is not None:
        d["n_heavy"] = m.GetNumHeavyAtoms()
        d["n_rings"] = rdMolDescriptors.CalcNumRings(m)
        d["n_arom_atoms"] = sum(a.GetIsAromatic() for a in m.GetAtoms())
        d["n_N"] = sum(a.GetSymbol() == "N" for a in m.GetAtoms())
        d["n_O"] = sum(a.GetSymbol() == "O" for a in m.GetAtoms())
        d["n_S"] = sum(a.GetSymbol() == "S" for a in m.GetAtoms())
        d["n_hal"] = sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms())
        d["n_hetero"] = sum(a.GetSymbol() in ("N", "O", "S", "P", "F", "Cl", "Br", "I") for a in m.GetAtoms())
        d["n_double_bonds"] = sum(1 for b in m.GetBonds() if b.GetBondTypeAsDouble() == 2.0)
        d["n_rot"] = rdMolDescriptors.CalcNumRotatableBonds(m)
    if formula:
        at = parse_formula(formula)
        c, h, n = at.get("C", 0), at.get("H", 0), at.get("N", 0)
        x = at.get("F", 0) + at.get("Cl", 0) + at.get("Br", 0) + at.get("I", 0)
        d["DBE"] = max((2 * c + 2 + n - h - x) / 2, 0)
    return d


DOC_PRIOR = {"enveda-np-examples": 4.0, "enveda-180": 2.0, "drug_plus": 1.0}
DOC_DEFAULT = 0.5


def build_pool():
    log("read train.parquet")
    tr = pd.read_parquet(f"{DATA}/train.parquet")

    log("dedup pool on inchikey14")
    pool = tr.groupby("inchikey14").agg(
        smiles=("normalized_smiles", "first"),
        formula=("molecular_formula", "first"),
        ingest_lib=("ingest_lib", "first"),
        n_rows=("ms2_mzs", "size"),
        n_smiles=("normalized_smiles", "nunique"),
    ).sort_index()
    pos_of_key = {k: i for i, k in enumerate(pool.index)}

    log("molecules: parse + mass + np features")
    pool["mol"] = pool["smiles"].map(Chem.MolFromSmiles)
    pool["mass"] = pool["mol"].map(lambda m: rdMolDescriptors.CalcExactMolWt(m) if m is not None else np.nan)
    feats = [np_features(m, f) for m, f in zip(pool["mol"], pool["formula"])]
    pool = pd.concat([pool, pd.DataFrame(feats, index=pool.index)], axis=1)
    pool["doc_prior"] = pool["ingest_lib"].map(lambda lib: DOC_PRIOR.get(lib, DOC_DEFAULT))

    log("molecule fingerprints (Morgan r2, 2048)")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    fp_bytes = []
    for m in pool["mol"]:
        v = np.zeros(FP_SIZE, dtype=np.uint8)
        f = gen.GetFingerprint(m)
        for b in f.GetOnBits():
            v[b] = 1
        fp_bytes.append(np.packbits(v).tobytes())
    pool["fp_bytes"] = fp_bytes

    log("library spectra (concatenated per molecule)")
    g = tr.groupby("inchikey14", sort=True)
    mzs, ints, offsets = [], [], [0]
    n_groups = 0
    for key, gg in g:
        if key not in pos_of_key:
            continue
        m = np.concatenate([np.asarray(a, dtype=np.float32) for a in gg.ms2_mzs])
        it = np.concatenate([np.asarray(a, dtype=np.float32) for a in gg.ms2_normalized_intensities])
        mzs.append(m)
        ints.append(it)
        offsets.append(offsets[-1] + len(m))
        n_groups += 1
    lib_mz = np.concatenate(mzs)
    lib_int = np.concatenate(ints)
    lib_offsets = np.array(offsets, dtype=np.int64)
    assert n_groups == len(pool), f"npz segments {n_groups} != pool size {len(pool)}"
    assert len(lib_offsets) == len(pool) + 1
    log(f"total library points: {len(lib_mz)} ({len(lib_mz) / len(pool):.0f}/molecule avg)")

    log("adduct offsets/tolerances from train (per-adduct Da, tight cap)")
    mass_of = pool["mass"]
    d_da = tr["precursor_mz"].to_numpy() - tr["inchikey14"].map(mass_of).to_numpy()
    adduct_stats = {}
    adducts = tr["adduct"].to_numpy()
    for a in tr["adduct"].unique():
        a = str(a)
        e = d_da[adducts == a]
        e = e[np.isfinite(e)]
        if len(e) == 0:
            continue
        shift = float(np.median(e))
        q = float(np.quantile(np.abs(e - shift), 0.995))
        tol = float(min(max(q, 0.005), 0.05))
        adduct_stats[a] = {"shift_da": shift, "tol_da": tol, "n": int(len(e))}
    e = d_da[np.isfinite(d_da)]
    adduct_stats["__default__"] = {
        "shift_da": float(np.median(e)),
        "tol_da": 0.02,
        "n": int(len(e)),
    }

    log("save pool.parquet / pool_spec.npz / adduct_offsets.json")
    cols = ["smiles", "formula", "mass", "ingest_lib", "doc_prior", "n_rows", "n_smiles",
            "n_rings", "n_arom_atoms", "n_N", "n_O", "n_S", "n_hal", "n_hetero", "n_heavy",
            "n_double_bonds", "n_rot", "DBE"]
    pool = pool.drop(columns=["mol"])
    pool.to_parquet(f"{DATA}/pool.parquet", index=True)
    np.savez_compressed(f"{DATA}/pool_spec.npz", lib_mz=lib_mz, lib_int=lib_int, lib_offsets=lib_offsets)
    with open(f"{DATA}/adduct_offsets.json", "w") as fh:
        json.dump(adduct_stats, fh, indent=1)

    log(f"pool: {len(pool)} molecules; mass range {float(np.nanmin(pool.mass)):.1f} to {float(np.nanmax(pool.mass)):.1f}")
    for k in sorted(adduct_stats, key=lambda k: -adduct_stats[k]["n"])[:8]:
        log(f"  {k}: n={adduct_stats[k]['n']}, shift={adduct_stats[k]['shift_da']:.4f} Da, tol={adduct_stats[k]['tol_da']:.4f} Da")


if __name__ == "__main__":
    sys.exit(build_pool())
