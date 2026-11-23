"""Debug: why does collect_records return 0 records? Check mass_candidates on real queries."""
import numpy as np
import pandas as pd

import pipeline as pl

ctx = pl.build_context()
tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
queries = pl.train_queries(tr, ctx)
del tr

stats = ctx["adduct_stats"]
for a in ("[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"):
    print(a, stats.get(a))

keys = list(queries)[:50]
for key in keys:
    q = queries[key]
    cands = pl.mass_candidates(ctx, q["prec"], q["adduct"], q["ionization_mode"])
    if not cands:
        print(f"key={key} prec={q['prec']:.3f} adduct={q['adduct']} -> 0 cands")
        m = ctx["col"]["mass"][ctx["pos_of_key"][key]]
        off = stats.get(q["adduct"], stats.get("__default__", {})).get("offset_ppm", 0)
        print(f"   mol_mass={m:.4f} mass_ppm_vs_neutral={(q['prec'] - m)/m*1e6:.3f} off={off}")
        if sum(1 for k in keys if k != key) > 3:
            break
print("sample queries ok?")
rec, report = pl.collect_records(ctx, {k: queries[k] for k in keys[:5]}, verbose=False)
print("records for 5 queries:", len(rec), report)
