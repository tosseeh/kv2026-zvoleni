"""Načtení open dat ČSÚ (KV 2026) do tabulky vstupů pro model."""

from pathlib import Path

import pandas as pd


def _gender(cand):
    first = cand.JMENO.str.strip().str.split().str[0].fillna("").str.capitalize()
    surname = cand.PRIJMENI.str.strip().str.split("-").str[0].str.split().str[-1].fillna("").str.lower()
    fem_surname = surname.str.endswith(("ová", "á"))
    stats = fem_surname.groupby(first).agg(["mean", "size"])
    fem_first = first.isin(stats[stats["mean"] > 0.5].index)
    return fem_first.where(first.map(stats["size"]) >= 5, fem_surname).astype(int)


def load_candidates(data_dir):
    """Ze složky s open daty ČSÚ vrátí tabulku kandidátů se sloupci model.INPUTS + OKRES a zvolen."""
    d = Path(data_dir)
    reg, cis = d / "KV2026reg20261011_csv" / "csv_od", d / "KV2026ciselniky20261011_csv" / "csv_od"
    kvrk = pd.read_csv(reg / "kvrk.csv", low_memory=False)
    kvros = pd.read_csv(reg / "kvros.csv")
    kvrzcoco = pd.read_csv(reg / "kvrzcoco.csv")
    kvcoco = pd.read_csv(cis / "kvcoco.csv")
    cvs = pd.read_csv(cis / "cvs.csv")

    lists = kvros.merge(cvs[["VSTRANA", "TYPVS"]], on="VSTRANA", how="left")
    held = kvcoco.loc[kvcoco.STAV_OBCE == 0, "KODZASTUP"]
    councils = (kvrzcoco.query("TYPZASTUP == 1 and OBVODY in [0, 1]")
                .drop_duplicates("KODZASTUP")[["KODZASTUP", "OKRES", "POCOBYV", "MANDATY"]])
    councils = councils[councils.KODZASTUP.isin(held)]
    councils["n_lists"] = councils.KODZASTUP.map(lists.groupby("KODZASTUP").size())

    cand = kvrk.query("PLATNOST == 'A'").merge(councils, on=["KODZASTUP", "OKRES"])
    cand = cand.merge(lists[["KODZASTUP", "COBVODU", "OSTRANA", "TYPVS"]], on=["KODZASTUP", "COBVODU", "OSTRANA"], how="left")

    return pd.DataFrame({
        "poradi": cand.PORCISLO,
        "mandaty": cand.MANDATY,
        "pocet_kandidatek": cand.n_lists,
        "pocet_kandidatu": cand.groupby("KODZASTUP").KODZASTUP.transform("size"),
        "obyvatel": cand.POCOBYV,
        "vek": cand.VEK,
        "zena": _gender(cand),
        "titul": (cand.TITULPRED.notna() | cand.TITULZA.notna()).astype(int),
        "starosta": cand.POVOLANI.str.contains("starost", case=False, na=False).astype(int),
        "typ": cand.TYPVS,
        "OKRES": cand.OKRES,
        "zvolen": cand.MANDAT.eq("A").astype(int),
    })
