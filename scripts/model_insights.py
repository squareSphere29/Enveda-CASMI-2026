"""Report the final blend model's feature importances (from data/model.pkl)."""
import pickle

import pipeline as pl

model = pickle.load(open("d:/Dev/KaggleEnvedaCASMI/data/model.pkl", "rb"))
try:
    booster = model.booster_
    names = booster.feature_name()
    imp = booster.feature_importance(importance_type="gain")
except Exception:
    booster = model.booster_
    names = booster.feature_name()
    imp = booster.feature_importance()
pairs = sorted(zip(names, imp), key=lambda t: -t[1])
for n, v in pairs[:15]:
    print(f"{n:28s} {v:10.1f}")
