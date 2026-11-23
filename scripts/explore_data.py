import pyarrow.parquet as pq

for f in ["train", "test"]:
    t = pq.read_table(f"d:/Dev/KaggleEnvedaCASMI/data/{f}.parquet")
    print(f"== {f}: {t.num_rows} rows")
    print(t.schema)
    print()
