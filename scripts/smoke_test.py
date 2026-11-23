"""Smoke test: run the pipeline on a few library molecules and check the plumbing."""
import pipeline as pl

ctx = pl.build_context()
off = ctx["adduct_stats"]["[M+H]+"]["shift_da"]
queries = {}
for key in list(ctx["pool"].index)[:5]:
    pos = ctx["pos_of_key"][key]
    lm, li = pl.spec_segment(ctx, key)
    queries[key] = {
        "q_mz": lm, "q_int": li,
        "prec": float(ctx["col"]["mass"][pos] * (1 + off / 1e6)),
        "adduct": "[M+H]+", "ionization_mode": "positive",
    }
recs = pl.collect_records(ctx, queries, verbose=False)
if not recs:
    print("no records - check mass matching")
    raise SystemExit(1)
for qid, q in queries.items():
    rs = [r for r in recs if r["qid"] == qid]
    truth_in = any(r["key"] == qid for r in rs)
    best = max(rs, key=lambda r: r["feats"]["spec_cos"])
    print(f"{qid}: truth_in_cands={truth_in} top1_is_truth={best['key'] == qid} "
          f"spec_cos_top={best['feats']['spec_cos']:.3f} n_cands={len(rs)} "
          f"v8={best['feats']['v8_rank']} anchored={best['feats']['anchored_rank']} "
          f"haideptry={best['feats']['haideptry_rank']}")
r0 = recs[0]
print("\nfeature keys:", sorted(r0["feats"].keys()))
print("sample feats:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in list(r0["feats"].items())[:12]})
