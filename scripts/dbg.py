from pathlib import Path

src = Path("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py").read_text(encoding="utf-8")
i = src.find("not in pool.columns")
print(repr(src[i - 80:i + 200]))
