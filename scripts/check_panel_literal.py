import re

import pandas as pd

src = open("d:/Dev/KaggleEnvedaCASMI/scripts/validate.py", encoding="utf-8").read()
lits = sorted(set(l for l in re.findall(r'"[^"]*enveda[^"]*"', src) if "np" in l))
print("validate.py np literals:", [(l, [hex(ord(c))[2:] for c in l][:8]) for l in lits])

tr = pd.read_parquet("d:/Dev/KaggleEnvedaCASMI/data/train.parquet")
vals = sorted(set(v for v in tr["ingest_lib"].unique() if "np" in str(v)))
print("train values with 'np':", [(v, [hex(ord(c))[2:] for c in v][:8]) for v in vals])
