from pathlib import Path

p = Path("d:/Dev/KaggleEnvedaCASMI/scripts/build_library.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
for i, l in enumerate(lines):
    if 'pool.to_parquet(f"{DATA}/pool.parquet", columns=cols)' in l:
        break
else:
    raise SystemExit("pattern not found")
print("found at line", i + 1, ":", lines[i - 1].strip(), "|", lines[i].strip())
new_block = [
    '    pool = pool.drop(columns=["mol"])\n',
    '    pool.to_parquet(f"{DATA}/pool.parquet", index=True)\n',
]
lines[i:i + 1] = new_block
p.write_text("".join(lines), encoding="utf-8")
print("patched")
