"""Restore np_likeness to pipeline.py."""
import ast
from pathlib import Path

p = Path("d:/Dev/KaggleEnvedaCASMI/scripts/pipeline.py")
src = p.read_text(encoding="utf-8")
assert "def np_likeness" not in src
fn = (
    "def np_likeness(row):\n"
    "    v = 0.0\n"
    '    v += 0.1 * min(row["n_rings"], 4)\n'
    '    v += 0.05 * min(row["n_arom_atoms"], 6) / 3\n'
    '    v += 0.1 * (row["n_N"] + row["n_S"]) / 2\n'
    '    v += 0.1 * min(row["n_O"], 4) / 4\n'
    '    v -= 0.3 * row["n_hal"] / max(row["n_heavy"], 1)\n'
    '    v -= 0.15 * row["n_rot"] / 20\n'
    "    return min(max(v, 0.0), 1.0)\n"
    "\n\n"
)
i = src.index("def ranker_v8")
src = src[:i] + fn + src[i:]
p.write_text(src, encoding="utf-8")
names = [n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]
print("np_likeness restored:", "np_likeness" in names)
