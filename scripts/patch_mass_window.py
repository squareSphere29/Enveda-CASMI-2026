"""Patch pipeline.mass_candidates to the Da-based adduct-shift window; update make_notebook MASS_MD."""
import ast
from pathlib import Path

p = Path("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py")
src = p.read_text(encoding="utf-8")

start = src.index("def mass_candidates")
end = src.index("def ranker_v8")
new_fn = '''def mass_candidates(ctx, prec, adduct, ionization_mode, max_hyps=4):
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
            if prev is None or abs(dev) < abs(prev[3]):
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


'''
src = src[:start] + new_fn + src[end:]
p.write_text(src, encoding="utf-8")
ast.parse(src)
print("pipeline.py mass_candidates patched")

m = Path("d:/Dev/KaggleEnvedaCASMI/scripts/make_notebook.py")
src = m.read_text(encoding="utf-8")
old_md = 'MASS_MD = """## 2. Mass matching: adduct-adjusted, offset-centered windows\n'
i = src.index(old_md)
j = src.index('"""', i + len(old_md)) + 3
new_md = 'MASS_MD = """## 2. Mass matching: adduct-adjusted, offset-centered windows\n\nFrom `train.parquet` we calibrated, per adduct, the physical shift of the observed\nprecursor vs the neutral mass (median of `precursor_mz - neutral_mass`: [M+H]+ +1.0076 Da,\n[M-H]- -1.0073 Da, [M+Na]+ 22.989 Da, [M+K]+ 38.963 Da, [M+NH4]+ 18.034 Da,\n[M+Cl]- 34.969 Da, the standard adduct masses) and the per-molecule tolerance\n(0.999 quantile of the molecule-level deviation + 0.5 Da, capped at 3 Da). The window on\nthe neutral mass is `|precursor - m - shift(adduct)| <= tol(adduct)`. We try the query\nadduct first and then other adducts of the same ionization mode, deduplicating\ncandidates by `inchokey14`.\n"""'
src = src[:i] + new_md + src[j:]
m.write_text(src, encoding="utf-8")
ast.parse(src)
print("make_notebook.py MASS_MD updated")
