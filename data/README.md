# Data

Data nejsou součástí repozitáře: kandidátní listiny obsahují jména a další osobní údaje.

Stáhni open data ČSÚ ke komunálním volbám 2026 z [volby.gov.cz](https://volby.gov.cz/opendata/kv2026/kv2026_opendata.htm)
(registr kandidátů a číselníky, formát CSV) a rozbal je sem tak, aby vznikly složky:

```
data/
├── KV2026reg20261011_csv/csv_od/    kvrk.csv, kvros.csv, kvrzcoco.csv
└── KV2026ciselniky20261011_csv/csv_od/    kvcoco.csv, cvs.csv
```

Pak `load_candidates("data")` vrátí tabulku kandidátů připravenou pro `predict()`.
