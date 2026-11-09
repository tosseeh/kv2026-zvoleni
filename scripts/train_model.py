"""Natrénuje model jako v Části II a uloží ho do balíčku.

Použití: python scripts/train_model.py <složka s daty ČSÚ> <test_okresy.csv>
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from kv2026_zvoleni import load_candidates, train

data_dir, test_file = sys.argv[1], sys.argv[2]
cand = load_candidates(data_dir)
test_okresy = pd.read_csv(test_file).OKRES

keep = ~cand.OKRES.isin(test_okresy) & (cand.pocet_kandidatu > cand.mandaty)
tr = cand[keep].reset_index(drop=True)
print(f"trénovací kandidáti: {len(tr):,}")

model = train(tr, tr.zvolen)
out = Path(__file__).parents[1] / "src" / "kv2026_zvoleni" / "model.joblib"
joblib.dump(model, out, compress=("xz", 9))
print("uloženo:", out)
