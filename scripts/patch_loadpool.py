from pathlib import Path

p = Path("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("def load_pool"))
end = next(i for i, l in enumerate(lines) if l.startswith("def build_fp_map"))
print("replacing lines", start + 1, "to", end)
new_func = '''def load_pool(data=DATA):
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


'''
lines[start:end] = [new_func]
p.write_text("".join(lines), encoding="utf-8")
print("patched")
