"""Plumbing check: adduct-shifted precursor, held-out-style query, truth in window."""
import pipeline as pl

ctx = pl.build_context()
off = float(ctx["adduct_stats"]["[M+H]+"]["shift_da"])
queries = {}
for key in list(ctx["pool"].index)[:5]:
    pos = ctx["pos_of_key"][key]
    lm, li = pl.spec_segment(ctx, key)
    queries[key] = {
        "q_mz": lm, "q_int": li,
        "prec": float(ctx["col"]["mass"][pos]) + off,
        "adduct": "[M+H]+", "ionization_mode": "positive",
    }
recs, report = pl.collect_records(ctx, queries, verbose=False)
print("mass miss", report["n_miss_mass"], "recall", round(report["recall_at_max"], 3))
if not recs:
    print("no records - check mass matching")
    raise SystemExit(1)
for qid in queries:
    rs = [r for r in recs if r["qid"] == str(qid)]
    truth_in = any(r["key"] == qid for r in rs)
    best = max(rs, key=lambda r: r["feats"]["spec_cos"]) if rs else None
    n = len(rs)
    if best is None:
        print(f"{qid}: no cands")
        continue
    print(f"{qid}: truth_in_cands={truth_in} top1_is_truth={best['key'] == qid} "
          f"spec_cos_top={best['feats']['spec_cos']:.3f} n_cands={n} "
          f"mass_ppm={best['feats']['mass_ppm']:.2f}")
