"""Assemble CASMI26_solution.ipynb from the tested pipeline code.

The notebook is self-contained: pipeline functions (from scripts/pipeline.py)
are inlined, and the validation / test-scoring drivers (scripts/validate.py,
scripts/score_test.py) are inlined with the module prefix removed.
"""
import json
from pathlib import Path

ROOT = Path("D:/Dev/KaggleEnvedaCASMI")
PIPE = (ROOT / "scripts/pipeline.py").read_text()
VAL = (ROOT / "scripts/validate.py").read_text()
SCORE = (ROOT / "scripts/score_test.py").read_text()
OUT = ROOT / "CASMI26_solution.ipynb"


def strip_module_prefix(src):
    src = src.replace("import pipeline as pl\n", "")
    src = src.replace("pl.", "")
    return src


def split_sections(src):
    parts, cur = [], []
    for line in src.splitlines(keepends=True):
        if line.startswith("# ==== "):
            if cur:
                parts.append("".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        parts.append("".join(cur))
    # drop empty sections (e.g., the trailing marker)
    def has_content(p):
        stripped = "\n".join(l for l in p.splitlines() if not l.startswith("# ==== ")).strip()
        return bool(stripped)
    return [p for p in parts if has_content(p)]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


INTRO_MD = """# CASMI26 - Molecule ID from Mass Spectra: solution notebook

Predict the 2D structure (up to 25 guesses per molecule) for the hidden test set
from the `test.parquet` spectra.

Method summary (the user asks are all covered below):

1. **Dedup on the metric key**: pool = train molecules deduplicated on `inchikey14`
   (the metric key: first 14 chars of the InChIKey of the tautomer-canonicalized
   molecule; verified against the column, see cell below).
2. **Tight mass matching**: adduct-adjusted Da windows, centered on the observed
   shift, same-mode monomer hypotheses, candidates deduped on `inchikey14`.
3. **Evidence channels blended**: spectral cosine, fingerprint (FP-net analog)
   match, in-silico fragment coverage, popularity, and natural-product /
   documentation priors; blended by a LightGBM ranker; two-plus public rankers
   (v8-style, anchored-network, haideptry-style) are integrated as features.
4. **Local validation**: identity-disjoint 5-fold CV over train (folds by
   `inchikey14`) plus the natural-product panel benchmark (`enveda-np-examples`,
   the closest library to the test set).
5. **Time limit**: the library build is cached in `data/` (`pool.parquet`,
   `pool_spec.npz`, `adduct_offsets.json`); reruns of this notebook do not
   recompute it unless the cache is deleted.

Output: `submission.csv` (columns `molecule_id`, `smiles`, up to 25 guesses
semicolon-separated).
"""

DEDUP_MD = """## 1. Metric key: dedup on `inchikey14`

The metric key is the first 14 characters of the InChIKey of the
tautomer-canonicalized molecule (`rdMolStandardize.TautomerEnumerator().Canonicalize`
+ `Chem.MolToInchiKey` -> `inchokey[:14]`). `train.parquet` ships the column
`inchikey14`, which we treat as the metric key; the cell below checks how often
the computed key matches the column. The candidate pool is the pool of
unique `inchikey14` values (one SMILES per key).
"""

MASS_MD = """## 2. Mass matching: adduct-adjusted, offset-centered windows

Per adduct, shift is the median of `precursor_mz - neutral_mass` on rows of that
adduct only. Tolerance is the 0.995 quantile of absolute residual, floored at
0.005 Da and capped at 0.05 Da. Extra hypotheses are same-mode monomers, not
dimers. `mass_ppm` is `(precursor - mass - shift) / mass * 1e6`.
"""

VALID_MD = """## 3. Local validation

The query is one held-out train spectrum. Cosine against the true key uses the
other rows of that molecule, not the concatenated library that still contains
the query peaks. Empty mass windows score 0. The np panel is held out of the
ranker training set.
"""

TEST_MD = """## 4. Test scoring

The pipeline runs on `test.parquet`: mass-candidate retrieval per molecule,
evidence channels, public-ranker features, LightGBM ranker predictions,
top-25 by predicted score -> `submission.csv`."""

RESULT_MD = """## 5. Results

Run the cells and fill in:

- fold MRRs (identity-disjoint CV): ...
- panel benchmark MRR (`enveda-np-examples`): ...
- submission: `submission.csv` (400 rows)."""

CONFIRM_CELL = '''
def confirm_metric_key():
    """Check: the metric key = InChIKey14 (tautomer canonicalized) == the train.parquet column."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    pool = load_pool()
    smiles = pool["smiles"].to_numpy()
    keys = np.array(pool.index)
    taut = rdMolStandardize.TautomerEnumerator()
    ok = bad = 0
    for key, smi in zip(keys[:2000], smiles[:2000]):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mc = taut.Canonicalize(m)
        ik = inchi.MolToInchiKey(mc)
        if ik[:14] == key:
            ok += 1
        else:
            bad += 1
    print(f"InChIKey14 (tautomer canonical) == inchikey14 column: ok={ok}, bad={bad}")


confirm_metric_key()
'''

SHOW_ADDUCT_CELL = '''
def show_adduct_stats():
    stats = json.load(open("data/adduct_offsets.json"))
    rows = sorted(
        ((a, v["shift_da"], v["tol_da"], v["n"]) for a, v in stats.items() if not a.startswith("__")),
        key=lambda r: -r[3])
    print(pd.DataFrame(rows, columns=["adduct", "shift_da", "tol_da", "n_train"]).to_string(index=False))


show_adduct_stats()
'''


def main():
    nb = {
        "cells": [
            md(INTRO_MD),
            *[code(s) for s in split_sections(PIPE)],
            md(DEDUP_MD),
            code(CONFIRM_CELL),
            md(MASS_MD),
            code(SHOW_ADDUCT_CELL),
            md(VALID_MD),
            code(strip_module_prefix(VAL)),
            md(TEST_MD),
            code(strip_module_prefix(SCORE)),
            md(RESULT_MD),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print("wrote", OUT, "cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
