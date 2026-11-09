# kv2026-zvoleni

Odhad pravděpodobnosti, že kandidát v komunálních volbách bude zvolen – jen z údajů, které jsou známé
**před volbami** (kandidátní listiny). Model je random forest natrénovaný na komunálních volbách 2026
(semestrální projekt 4EK481, VŠE).

**Pro koho:** strany a sdružení při skládání kandidátky, novináři a analytici pro předvolební odhady.

## Instalace

```bash
pip install "git+https://github.com/tosseeh/kv2026-zvoleni@v0.1.0"
```

Python 3.11+.

## Použití

```python
import pandas as pd
from kv2026_zvoleni import predict

kandidati = pd.DataFrame([
    dict(poradi=1, mandaty=9, pocet_kandidatek=2, pocet_kandidatu=18, obyvatel=450,
         vek=52, zena=0, titul=0, starosta=1, typ="M"),
])
predict(kandidati)   # array([0.998])
```

| Sloupec | Význam |
|---|---|
| `poradi` | pořadí na kandidátce |
| `mandaty` | počet mandátů v zastupitelstvu |
| `pocet_kandidatek` | počet kandidátek v obci |
| `pocet_kandidatu` | počet všech kandidátů v obci |
| `obyvatel` | počet obyvatel obce |
| `vek` | věk kandidáta (min. 18) |
| `zena`, `titul`, `starosta` | 0/1 (starosta = má „starosta/místostarosta“ v povolání) |
| `typ` | typ kandidátky: `S` strana, `K` koalice, `D` strana + nezávislí, `M` sdružení nezávislých, `N` nezávislý kandidát |

Chybějící nebo nesmyslné vstupy (neznámý typ, pořadí 0, věk pod 18) funkce odmítne s chybou;
hodnoty mimo rozsah trénovacích dat projdou s varováním.

Celou obec z open dat ČSÚ připraví `load_candidates("data")` – viz [data/README.md](data/README.md).

## Jak dobrý model je

Cross-validace po okresech (62 okresů, 129 949 kandidátů): log loss 0,340, AUC 0,91; pravděpodobnosti jsou
kalibrované (z kandidátů s odhadem 80 % je zvoleno zhruba 80 %). Baseline jen s pořadím: log loss 0,434.

**Omezení:** natrénováno jen na volbách 2026 – na budoucí volby (nové strany, jiná nálada) se přenáší jen
s rezervou. Neplatí pro obce, kde je kandidátů méně nebo stejně jako mandátů (zvolen je každý).

## Testy

```bash
pip install ".[test]"
pytest
```

Testy běží automaticky v GitHub Actions po každé změně: stejné výsledky jako model z reportu, žádné údaje
z výsledků voleb na vstupu, odmítnutí špatných vstupů a reprodukovatelnost.

## Přetrénování

```bash
python scripts/train_model.py data test_okresy.csv
```
