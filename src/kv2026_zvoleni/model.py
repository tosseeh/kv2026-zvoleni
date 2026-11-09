"""Pravděpodobnost zvolení kandidáta v komunálních volbách z údajů známých před volbami."""

import warnings
from importlib import resources

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUTS = ["poradi", "mandaty", "pocet_kandidatek", "pocet_kandidatu", "obyvatel", "vek", "zena", "titul", "starosta", "typ"]
TYPES = ["D", "K", "M", "N", "S"]
FEATURES = ["PORCISLO", "rel_pos", "n_lists", "cand_per_seat", "log_pop", "VEK", "woman", "title", "starosta"]
TRAIN_RANGE = {"obyvatel": (27, 1_407_084), "mandaty": (5, 65)}
SEED = 42


def prepare(df):
    """Zkontroluje vstupy a spočítá featury modelu."""
    missing = [c for c in INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")
    x = df[INPUTS]
    if x.isna().any().any():
        raise ValueError(f"Chybějící hodnoty ve sloupcích: {list(x.columns[x.isna().any()])}")

    unknown = sorted(set(x.typ) - set(TYPES))
    if unknown:
        raise ValueError(f"Neznámý typ kandidátky {unknown}, povolené jsou {TYPES}")
    for c in ["poradi", "mandaty", "pocet_kandidatek", "pocet_kandidatu", "obyvatel"]:
        if (x[c] < 1).any():
            raise ValueError(f"Sloupec {c} musí být alespoň 1")
    if (x.vek < 18).any():
        raise ValueError("Kandidát musí mít alespoň 18 let")
    for c in ["zena", "titul", "starosta"]:
        if not x[c].isin([0, 1]).all():
            raise ValueError(f"Sloupec {c} musí být 0 nebo 1")

    for c, (lo, hi) in TRAIN_RANGE.items():
        out = (x[c] < lo) | (x[c] > hi)
        if out.any():
            warnings.warn(f"{out.sum()} řádků má {c} mimo rozsah trénovacích dat ({lo}–{hi}), predikce může být nespolehlivá")
    if (x.pocet_kandidatu <= x.mandaty).any():
        warnings.warn("Některé obce nemají víc kandidátů než mandátů – zvolen je každý, model tyto obce neviděl")

    return pd.DataFrame({
        "PORCISLO": x.poradi,
        "rel_pos": x.poradi / x.mandaty,
        "n_lists": x.pocet_kandidatek,
        "cand_per_seat": x.pocet_kandidatu / x.mandaty,
        "log_pop": np.log10(x.obyvatel),
        "VEK": x.vek,
        "woman": x.zena,
        "title": x.titul,
        "starosta": x.starosta,
        "TYPVS": x.typ,
    })


def make_model(seed=SEED):
    """Random forest se stejným nastavením jako v Části II."""
    prep = ColumnTransformer([("num", "passthrough", FEATURES),
                              ("typ", OneHotEncoder(handle_unknown="ignore"), ["TYPVS"])])
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_features=0.6, n_jobs=-1, random_state=seed)
    return Pipeline([("prep", prep), ("clf", rf)])


def train(df, y, seed=SEED):
    """Natrénuje model na tabulce vstupů (INPUTS) a vektoru zvolen 0/1."""
    return make_model(seed).fit(prepare(df), y)


def load_model():
    """Načte model, který je součástí balíčku."""
    with resources.as_file(resources.files("kv2026_zvoleni") / "model.joblib") as path:
        return joblib.load(path)


def predict(df, model=None):
    """Vrátí pravděpodobnost zvolení pro každý řádek."""
    model = model or load_model()
    model.set_params(clf__n_jobs=1)  # paralelní sčítání stromů by dávalo rozdíly v posledních desetinných místech
    return model.predict_proba(prepare(df))[:, 1]
