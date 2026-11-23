"""CASMI26 solution pipeline: candidate retrieval by mass + reranking on evidence channels.

Channels per candidate (used by the public rankers and the LightGBM ranker):
  mass:        adduct-adjusted, offset-centered mass window (calibrated from train)
  spec_cos:    cosine of the query spectrum vs the molecule's library spectra
  fp_match:    Morgan-FP Tanimoto of the candidate vs the spectral winner
  in_silico:   fraction of observed fragments explained by the candidate's
               theoretical (single-bond cleavage) fragments
  popularity:  training-set spectrum count of the molecule
  doc_prior:   ingest-lib weight (natural-product / documentation prior)
  np features: natural-product-flavoured descriptors (rings, aromaticity, N/O/S, ...)
"""

import json
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

DATA = "d:/Dev/KaggleEnvedaCASMI/data"
FP_SIZE = 2048
PROTON = 1.007276

COLS = ["mass", "n_rows", "doc_prior", "ingest_lib", "n_rings", "n_arom_atoms", "n_N",
        "n_O", "n_S", "n_hal", "n_hetero", "n_heavy", "n_double_bonds", "n_rot", "DBE"]

FEATURE_NAMES = ["mass_match", "mass_ppm", "mass_abs_ppm", "spec_cos", "fp_match",
                 "fp_mean_top5", "in_silico", "popularity", "doc_prior", "n_rings",
                 "n_arom_atoms", "n_N", "n_O", "n_S", "n_hal", "n_hetero", "n_heavy",
                 "n_double_bonds", "n_rot", "DBE", "idx", "v8_score", "v8_rank",
                 "anchored_score", "anchored_rank", "haideptry_score", "haideptry_rank",
                 "pub_avg_rank"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- loading

# ==== loading

def load_pool(data=DATA):
    pool = pd.read_parquet(f"{data}/pool.parquet")
    if "inchikey14" not in pool.columns:
        if pool.index.name == "inchikey14":
            pool = pool.reset_index()
        elif pool.index.name is not None:
            pool = pool.reset_index().rename(columns={pool.index.name: "inchikey14"})
        else:
            pool = pool.reset_index(names="inchikey14")
    pool = pool.set_index("inchikey14").sort_index()
    pool["mol"] = pool["smiles"].map(Chem.MolFromSmiles)
    return pool


def build_fp_map(pool):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    out = {}
    for k, m in zip(pool.index, pool["mol"]):
        v = np.zeros(FP_SIZE, dtype=np.uint8)
        if m is not None:
            f = gen.GetFingerprint(m)
            idx = np.fromiter(f.GetOnBits(), dtype=np.int32, count=f.GetNumOnBits())
            if len(idx):
                v[idx] = 1
        out[k] = np.packbits(v).tobytes()
    return out


def build_context(data=DATA):
    log("load pool")
    pool = load_pool(data)
    log("sort pool by mass")
    pool_by_mass = pool.sort_values("mass", kind="stable", na_position="last")
    mass_v = pool_by_mass["mass"].to_numpy()
    valid = np.isfinite(mass_v)
    mass_v = mass_v[valid]
    keys_v = pool_by_mass.index.to_numpy()[valid]
    pos_of_key = {k: i for i, k in enumerate(pool.index)}
    log("load spec arrays")
    z = np.load(f"{data}/pool_spec.npz")
    with open(f"{data}/adduct_offsets.json") as fh:
        adduct_stats = json.load(fh)
    col = {c: pool[c].to_numpy() for c in COLS}
    smiles = pool["smiles"].to_numpy()
    log("build Morgan fingerprints (all molecules)")
    fp_map = build_fp_map(pool)
    return {
        "pool": pool, "pool_by_mass": pool_by_mass, "mass_v": mass_v, "keys_v": keys_v,
        "pos_of_key": pos_of_key, "lib_mz": z["lib_mz"], "lib_int": z["lib_int"],
        "lib_offsets": z["lib_offsets"], "adduct_stats": adduct_stats, "col": col,
        "smiles": smiles, "fp_map": fp_map, "theo_cache": {},
    }


def spec_segment(ctx, key):
    pos = ctx["pos_of_key"][key]
    off = ctx["lib_offsets"]
    mz = ctx["lib_mz"][off[pos]:off[pos + 1]]
    it = ctx["lib_int"][off[pos]:off[pos + 1]]
    o = np.argsort(mz, kind="stable")
    return mz[o], it[o]


# ---------------------------------------------------------------- spectrum utils

# ==== spectrum utils

def spec_cos(q_mz, q_int, lm, li):
    """Cosine between two spectra; each library peak matches the nearest query
    peak within tol."""
    if len(lm) == 0 or len(q_mz) == 0:
        return 0.0
    qv = q_int / (np.linalg.norm(q_int) + 1e-12)
    lv = li / (np.linalg.norm(li) + 1e-12)
    s = np.searchsorted(q_mz, lm)
    a = np.clip(s - 1, 0, len(q_mz) - 1)
    b = np.clip(s, 0, len(q_mz) - 1)
    da = np.abs(q_mz[a] - lm)
    db = np.abs(q_mz[b] - lm)
    idx = np.where(da <= db, a, b)
    tol = 0.02 + 0.0015 * np.abs(lm)
    m = np.minimum(da, db) <= tol
    if not m.any():
        return 0.0
    return float(np.dot(qv[idx[m]], lv[m]))


def theoretical_fragments(mol):
    """Neutral masses of single-bond-cleavage fragments of the molecule.

    BFS per bond: component of the cut endpoint (the other fragment is the
    remainder of the molecule).
    """
    if mol is None:
        return np.array([])
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    amass = np.zeros(n)
    ah = np.zeros(n)
    for i, a in enumerate(mol.GetAtoms()):
        amass[i] = a.GetMass()
        ah[i] = a.GetTotalNumHs()
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[a1].append(a2)
        adj[a2].append(a1)
    comp_mass = amass + ah * 1.007825
    total = float(comp_mass.sum())
    frags = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        seen = np.zeros(n, dtype=bool)
        seen[a1] = True
        stack = [a1]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if (x == a1 and y == a2) or (x == a2 and y == a1):
                    continue
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        frags.append(float(comp_mass[seen].sum()))
        frags.append(total - frags[-1])
    return np.array(frags, dtype=np.float64)


def in_silico(ctx, key, q_mz, ionization_mode="positive"):
    """Fraction of theoretical fragments of the candidate observed in the query spectrum."""
    theo = ctx["theo_cache"].get(key)
    if theo is None:
        cap = ctx.get("theo_cap")
        if cap is not None and len(ctx["theo_cache"]) >= cap:
            theo = np.array([])
        else:
            m = Chem.MolFromSmiles(ctx["smiles"][ctx["pos_of_key"][key]])
            theo = theoretical_fragments(m)
        ctx["theo_cache"][key] = theo
    if len(theo) == 0 or len(q_mz) == 0:
        return 0.0
    shift = PROTON if ionization_mode == "positive" else -PROTON
    exp = theo + shift
    obs = np.asarray(q_mz, dtype=np.float64)
    idx = np.searchsorted(obs, exp)
    left = idx - 1
    in_r = (idx >= 0) & (idx < len(obs))
    d_right = np.where(in_r, np.abs(obs[np.clip(idx, 0, len(obs) - 1)] - exp), np.inf)
    in_l = (left >= 0) & (left < len(obs))
    d_left = np.where(in_l, np.abs(obs[np.clip(left, 0, len(obs) - 1)] - exp), np.inf)
    found = np.sum(np.minimum(d_left, d_right) <= 0.03 + 0.001 * np.abs(exp))
    return float(found / len(exp))


def fp_tanimoto(b1, b2):
    a = np.unpackbits(np.frombuffer(b1, dtype=np.uint8))
    b = np.unpackbits(np.frombuffer(b2, dtype=np.uint8))
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------- mass matching

# ==== mass matching

def mass_candidates(ctx, prec, adduct, ionization_mode, max_hyps=4):
    """Candidates whose observed precursor falls in the adduct-adjusted window:
    |prec - m - shift[a]| <= tol[a] (shift/tol calibrated from train). Returns
    {key: (key, mass, dev, hyp, mass_ppm)}."""
    mass, keys = ctx["mass_v"], ctx["keys_v"]
    stats = ctx["adduct_stats"]
    hyps = []
    if adduct in stats:
        hyps.append(adduct)
    for a in sorted((a for a in stats if not a.startswith("__") and a != adduct),
                    key=lambda a: -stats[a].get("n", 0)):
        hyps.append(a)
    if not hyps:
        hyps = [adduct]
    out = {}
    for a in hyps[:max_hyps]:
        st = stats.get(a, stats.get("__default__", {"shift_da": 0.0, "tol_da": 3.0}))
        shift, tol = st["shift_da"], max(0.05, st["tol_da"])
        target = prec - shift
        i0 = np.searchsorted(mass, target - tol, side="left")
        i1 = np.searchsorted(mass, target + tol, side="left")
        for j in range(i0, i1):
            key = keys[j]
            m = mass[j]
            if not np.isfinite(m):
                continue
            dev = prec - m - shift
            prev = out.get(key)
            if prev is None or abs(dev) < abs(prev[2]):
                out[key] = (key, m, dev, a, (prec - m) / m * 1e6)
    if not out:
        # fallback: relax around the adduct-adjusted precursor
        st = stats.get(adduct, stats.get("__default__", {"shift_da": 0.0, "tol_da": 3.0}))
        target = prec - st["shift_da"]
        near = np.where(np.abs(mass - target) <= max(st["tol_da"], 0.05) * 10)[0]
        if len(near):
            for j in near[np.argsort(np.abs(mass[near] - target))[:5]]:
                m = mass[j]
                out[keys[j]] = (keys[j], m, prec - m - st["shift_da"], adduct,
                                 (prec - m) / m * 1e6)
    return out


def np_likeness(row):
    v = 0.0
    v += 0.1 * min(row["n_rings"], 4)
    v += 0.05 * min(row["n_arom_atoms"], 6) / 3
    v += 0.1 * (row["n_N"] + row["n_S"]) / 2
    v += 0.1 * min(row["n_O"], 4) / 4
    v -= 0.3 * row["n_hal"] / max(row["n_heavy"], 1)
    v -= 0.15 * row["n_rot"] / 20
    return min(max(v, 0.0), 1.0)


def ranker_v8(cand_list):
    """Analog of the mathleteutd v8 mass rule + isomer rerank."""
    scored = []
    for c in cand_list:
        sc = c["spec_cos"] - 0.15 * abs(c["mass_ppm"]) + 0.5 * c["mass_match"]
        scored.append((sc, c))
    if scored:
        top = min(scored, key=lambda t: abs(t[1]["mass_ppm"]))[1]
        if abs(top["mass_ppm"]) > 5 and top["spec_cos"] < 0.9:
            scored = [(sc * 0.7 if c is top else sc, c) for sc, c in scored]
    scored.sort(key=lambda t: -t[0])
    return {c["key"]: (sc, i) for i, (sc, c) in enumerate(scored)}


def doc_prior(ctx, key):
    """Documentation / natural-product prior: the np-examples panel gets the top
    weight; other ingested libraries their own weights."""
    if key in ctx.get("np_set", ()):
        return 4.0
    return float(ctx["col"]["doc_prior"][ctx["pos_of_key"][key]])


def ranker_anchored(ctx, cand_list):
    """Analog of the flexonafft anchored network: POP/SPEC/NP weights (0.12 each),
    LIB_HIT_GATE = 0.78, K14 dedup (candidate set is keyed on inchikey14)."""
    scored = []
    for c in cand_list:
        r = c["row"]
        sc = 0.12 * np.log10(r["n_rows"] + 1) + 0.12 * c["spec_cos"] + 0.12 * np_likeness(r)
        sc += 0.12 * doc_prior(ctx, c["key"])
        sc += 0.12 if c["spec_cos"] >= 0.78 else 0.0   # LIB_HIT_GATE
        sc += 0.12 if c["mass_match"] else 0.0
        scored.append((sc, c))
    scored.sort(key=lambda t: -t[0])
    return {c["key"]: (sc, i) for i, (sc, c) in enumerate(scored)}


def ranker_haideptry(ctx, cand_list):
    """Analog of the haideptry starter: retrieval + rerank across four channels
    (mass, fingerprint, spectral cosine, popularity)."""
    scored = []
    for c in cand_list:
        r = c["row"]
        sc = 0.25 * c["mass_match"] + 0.35 * c["spec_cos"] + 0.15 * c["fp_match"] \
            + 0.25 * np.log10(r["n_rows"] + 1) / np.log10(10001)
        scored.append((sc, c))
    scored.sort(key=lambda t: -t[0])
    return {c["key"]: (sc, i) for i, (sc, c) in enumerate(scored)}


# ---------------------------------------------------------------- candidates + features

# ==== candidates + features

def cand_features(ctx, cand, proxy_key, top5_keys):
    r = cand["row"]
    f = {
        "mass_match": float(cand["mass_match"]),
        "mass_ppm": cand["mass_ppm"],
        "mass_abs_ppm": abs(cand["mass_ppm"]),
        "spec_cos": cand["spec_cos"],
        "fp_match": cand["fp_match"],
        "fp_mean_top5": cand["fp_mean_top5"],
        "in_silico": cand["in_silico"],
        "popularity": float(r["n_rows"]),
        "doc_prior": doc_prior(ctx, cand["key"]),
        "n_rings": float(r["n_rings"]),
        "n_arom_atoms": float(r["n_arom_atoms"]),
        "n_N": float(r["n_N"]), "n_O": float(r["n_O"]), "n_S": float(r["n_S"]),
        "n_hal": float(r["n_hal"]), "n_hetero": float(r["n_hetero"]),
        "n_heavy": float(r["n_heavy"]), "n_double_bonds": float(r["n_double_bonds"]),
        "n_rot": float(r["n_rot"]), "DBE": float(r["DBE"]),
        "idx": float(cand["idx"]),
    }
    for name in ("v8", "anchored", "haideptry"):
        if name in cand["public"]:
            sc, rank = cand["public"][name]
            f[f"{name}_score"] = float(sc)
            f[f"{name}_rank"] = float(rank)
    avail = [f[n] for n in FEATURE_NAMES if n in f and f[n] is not None]
    if avail:
        f["pub_avg_rank"] = float(np.mean([f[n] for n in ("v8_rank", "anchored_rank", "haideptry_rank") if n in f]))
    for n in FEATURE_NAMES:
        if n not in f:
            f[n] = 0.0
    return f


def collect_records(ctx, queries, max_cands=25, verbose=True):
    """For each query molecule: mass candidates, evidence channels, public ranker
    lists, and a feature row. Returns list of records:
    {qid, key, label, feats} (one per candidate)."""
    records = []
    for qi, (qid, q) in enumerate(sorted(queries.items(), key=lambda kv: str(kv[0]))):
        cands = mass_candidates(ctx, q["prec"], q["adduct"], q["ionization_mode"])
        if not cands:
            continue
        cand_list = []
        for idx, (key, (k, m, err, hyp, ppm)) in enumerate(sorted(cands.items(), key=lambda kv: str(kv[0]))):
            lm, li = spec_segment(ctx, key)
            sc = spec_cos(q["q_mz"], q["q_int"], lm, li)
            cand_list.append({"key": key, "mass": m, "mass_ppm": ppm, "err": err,
                              "hyp": hyp, "mass_match": 1.0, "spec_cos": sc, "idx": idx})
        if not cand_list:
            continue
        cand_list.sort(key=lambda c: (-c["spec_cos"], c["key"]))
        proxy_key = cand_list[0]["key"]
        top5_keys = [c["key"] for c in cand_list[:5]]
        fp_map = ctx["fp_map"]
        for c in cand_list:
            pos = ctx["pos_of_key"][c["key"]]
            c["row"] = {name: ctx["col"][name][pos] for name in COLS}
            c["fp_match"] = fp_tanimoto(fp_map[c["key"]], fp_map[proxy_key])
            c["fp_mean_top5"] = float(np.mean([fp_tanimoto(fp_map[c["key"]], fp_map[t]) for t in top5_keys]))
            c["in_silico"] = (in_silico(ctx, c["key"], q["q_mz"], q["ionization_mode"])
                              if c["spec_cos"] >= 0.3 else 0.0)
        v8 = ranker_v8(cand_list)
        anchored = ranker_anchored(ctx, cand_list)
        hai = ranker_haideptry(ctx, cand_list)
        for c in cand_list:
            c["public"] = {"v8": v8[c["key"]], "anchored": anchored[c["key"]],
                           "haideptry": hai[c["key"]]}
        for c in cand_list[:max_cands]:
            records.append({"qid": str(qid), "key": c["key"],
                            "label": 1.0 if c["key"] == str(qid) else 0.0,
                            "feats": cand_features(ctx, c, proxy_key, top5_keys)})
        if verbose and qi % 10000 == 0:
            log(f"collect_records: {qi}/{len(queries)} queries, {len(records)} records")
    records.sort(key=lambda r: r["qid"])
    return records


def feats_array(records):
    return np.array([[r["feats"][n] for n in FEATURE_NAMES] for r in records], dtype=np.float64)


def mrr(records, preds):
    """Mean reciprocal rank of the truth candidate (label 1) within the
    per-query prediction order."""
    by_q = defaultdict(list)
    for r, p in zip(records, preds):
        by_q[r["qid"]].append((p, r["label"], r["key"]))
    rr = []
    for qid, lst in by_q.items():
        truth = [k for (p, l, k) in lst if l == 1]
        if not truth:
            rr.append(0.0)
            continue
        ranked = sorted(lst, key=lambda t: (-t[0], t[2]))
        for pos, (p, l, k) in enumerate(ranked):
            if k in truth:
                rr.append(1.0 / (pos + 1))
                break
        else:
            rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0


def make_folds(keys, n_folds=5, seed=42):
    keys = sorted(map(str, keys))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    folds = [set() for _ in range(n_folds)]
    for i, k in enumerate(perm):
        folds[i % n_folds].add(keys[k])
    return folds


def fit_predict_fold(Xtr, ytr, qids_tr, Xte):
    codes = pd.factorize(qids_tr, sort=False)[0]
    group = np.bincount(codes)
    model = lgb.LGBMRanker(n_estimators=250, learning_rate=0.05, num_leaves=31, verbose=-1)
    model.fit(Xtr, ytr, group=group)
    return model.predict(Xte)


def train_final(X, y, qids):
    codes = pd.factorize(qids, sort=False)[0]
    group = np.bincount(codes)
    model = lgb.LGBMRanker(n_estimators=250, learning_rate=0.05, num_leaves=31, verbose=-1)
    model.fit(X, y, group=group)
    return model


# ---------------------------------------------------------------- query tables

# ==== query tables

def train_queries(train_df, ctx):
    """Per molecule key: query spectrum (merged train rows), median precursor,
    adduct of the median-precursor row, first ionization mode."""
    key_col = next(c for c in train_df.columns if "key14" in c)
    g = train_df.groupby(key_col)
    prec_map = g["precursor_mz"].median()
    mode_map = g["ionization_mode"].first()

    def med_adduct(d):
        i = d["precursor_mz"].sub(np.median(d["precursor_mz"])).abs().idxmin()
        return str(d.at[i, "adduct"])

    adduct_map = g.apply(med_adduct, include_groups=False)
    queries = {}
    for key in ctx["pool"].index:
        if key not in prec_map.index:
            continue
        lm, li = spec_segment(ctx, key)
        queries[str(key)] = {
            "q_mz": lm, "q_int": li, "prec": float(prec_map[key]),
            "adduct": str(adduct_map[key]), "ionization_mode": str(mode_map[key]),
        }
    return queries


def test_queries(test_df):
    q = {}
    for mid, sub in test_df.groupby("molecule_id"):
        q_mz = np.concatenate([np.asarray(a, np.float64) for a in sub.ms2_mzs])
        q_int = np.concatenate([np.asarray(a, np.float64) for a in sub.ms2_normalized_intensities])
        o = np.argsort(q_mz, kind="stable")
        q_mz, q_int = q_mz[o], q_int[o]
        med = float(np.median(sub["precursor_mz"]))
        i = sub["precursor_mz"].sub(med).abs().idxmin()
        q[str(mid)] = {
            "q_mz": q_mz, "q_int": q_int, "prec": med,
            "adduct": str(sub.at[i, "adduct"]),
            "ionization_mode": str(sub.ionization_mode.iloc[0]),
        }
    return q


def submission_rows(ctx, records, preds, max_len=25):
    """[(molecule_id, 'smiles;smiles;...')] per query, top-max_len candidates."""
    by_q = defaultdict(list)
    for r, p in zip(records, preds):
        by_q[r["qid"]].append((p, r["key"]))
    rows = []
    for qid in sorted(by_q, key=str):
        ranked = sorted(by_q[qid], key=lambda t: (-t[0], t[1]))
        keys = [k for _, k in ranked[:max_len]]
        smiles = ";".join(ctx["smiles"][ctx["pos_of_key"][k]] for k in keys)
        rows.append((qid, smiles))
    return rows


# ==== (end)
