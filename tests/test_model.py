from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kv2026_zvoleni import INPUTS, predict, train
from kv2026_zvoleni.model import FEATURES

CASES = pd.read_csv(Path(__file__).parent / "cases.csv")
X = CASES[INPUTS]


# 1. Stejné odpovědi jako model z Části II
def test_same_answers_as_part2_model():
    np.testing.assert_allclose(predict(X), CASES.ocekavana_p, atol=1e-8)


# 2. Jen údaje známé před volbami
def test_no_post_election_features():
    forbidden = {"POCHLASU", "PROCHLASU", "MANDAT", "PORADIMAND", "PORADINAHR", "PRESKOCENI", "HLASY_STR", "zvolen"}
    assert not forbidden & set(FEATURES)


def test_answer_in_input_is_ignored():
    leak = X.assign(zvolen=1, POCHLASU=9999)
    np.testing.assert_array_equal(predict(leak), predict(X))


# 3. Špatné vstupy
@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns="typ"), "Chybí sloupce"),
    (lambda d: d.assign(vek=np.nan), "Chybějící hodnoty"),
    (lambda d: d.assign(typ="X"), "Neznámý typ"),
    (lambda d: d.assign(poradi=0), "alespoň 1"),
    (lambda d: d.assign(vek=16), "18 let"),
    (lambda d: d.assign(zena=2), "0 nebo 1"),
])
def test_bad_inputs_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        predict(change(X.copy()))


def test_out_of_range_warns():
    with pytest.warns(UserWarning, match="mimo rozsah"):
        predict(X.assign(obyvatel=5_000_000))


# 4. Reprodukovatelnost
def test_same_input_same_output():
    np.testing.assert_array_equal(predict(X), predict(X))


def test_training_is_reproducible():
    rng = np.random.default_rng(0)
    n = 1000
    d = pd.DataFrame({
        "poradi": rng.integers(1, 15, n), "mandaty": 15, "pocet_kandidatek": rng.integers(2, 6, n),
        "pocet_kandidatu": 60, "obyvatel": rng.integers(300, 20000, n), "vek": rng.integers(18, 80, n),
        "zena": rng.integers(0, 2, n), "titul": rng.integers(0, 2, n), "starosta": rng.integers(0, 2, n),
        "typ": rng.choice(["S", "M", "N"], n),
    })
    y = (d.poradi < 6).astype(int)
    p1 = predict(d, train(d, y))
    p2 = predict(d, train(d, y))
    np.testing.assert_array_equal(p1, p2)
