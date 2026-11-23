from pathlib import Path

for f in sorted(Path("d:/Dev/KaggleEnvedaCASMI/scripts").glob("*.py")):
    src = f.read_text(encoding="utf-8")
    n = src.count("inchikey14")
    if n:
        src = src.replace("inchikey14", "inchikey14")
        f.write_text(src, encoding="utf-8")
        print(f.name, "replaced", n)
print("done")
