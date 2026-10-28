"""Small generated tables to try regression and forecasting offline (always the same, seed 42).

house-prices   regress:  price of a house from size, rooms, age, district (1,500 rows)
shop-sales     forecast: daily sales of 3 shops over 2 years, trend + weekly / yearly season
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _house_prices(rng: np.random.Generator) -> pd.DataFrame:
    n = 1500
    district = rng.choice(
        ["centre", "suburb", "riverside", "industrial"], n, p=[0.2, 0.45, 0.15, 0.2]
    )
    size = rng.normal(120, 40, n).clip(30, 400).round()
    rooms = (size / 35 + rng.normal(0, 0.8, n)).clip(1, 8).round()
    age = rng.integers(0, 60, n)
    garage = rng.random(n) < 0.55
    base = {"centre": 95_000, "suburb": 55_000, "riverside": 80_000, "industrial": 38_000}
    price = (
        np.array([base[d] for d in district]) * size / 100
        + rooms * 9_000
        - age * 1_100
        + garage * 25_000
        + rng.normal(0, 18_000, n)
    )
    df = pd.DataFrame(
        {
            "size_m2": size,
            "rooms": rooms,
            "age_years": age,
            "garage": garage,
            "district": district,
            "price": price.clip(20_000).round(-2),
        }
    )
    df.loc[rng.random(n) < 0.03, "age_years"] = np.nan  # a few gaps, like real data
    return df


def _shop_sales(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(len(days))
    rows = []
    for shop, level in (("bangkok", 520), ("chiang-mai", 260), ("phuket", 340)):
        weekly = np.array([0.9, 0.85, 0.9, 0.95, 1.1, 1.35, 1.25])[days.dayofweek]
        yearly = 1 + 0.18 * np.sin(2 * np.pi * (days.dayofyear - 80) / 365.25)
        promo = rng.random(len(days)) < 0.06
        sales = level * (1 + t / 1500) * weekly * yearly * (1 + 0.35 * promo) + rng.normal(
            0, level * 0.05, len(days)
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": days.strftime("%Y-%m-%d"),
                    "shop": shop,
                    "promotion": promo.astype(int),
                    "sales": sales.round(1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


SAMPLES = {"house-prices": _house_prices, "shop-sales": _shop_sales}


def make(name: str) -> pd.DataFrame:
    if name not in SAMPLES:
        raise KeyError(f"Unknown table sample {name!r}: {list(SAMPLES)}")
    return SAMPLES[name](np.random.default_rng(42))
