import re
from pathlib import Path

import pandas as pd

real = [c for c in pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet").columns if "key14" in c][0]
print("REAL:", repr(real))
print()
for f in sorted(Path("d:/Dev/KaggleEnvedaCASMI/scripts").glob("*.py")):
    src = f.read_text(encoding="utf-8")
    lits = sorted(set(re.findall(r'["\'][^"\']*inch[^"\']*["\']', src)))
    if not lits:
        continue
    print(f.name + ":")
    for l in lits:
        name = l[1:-1]
        print("   ", repr(name), "match_real:", name == real)
