"""
Synthetic data generator that replicates the BCG PowerCo dataset schema.
Produces statistically realistic client and price data for the churn analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_CLIENTS = 14606
N_CHURNED = int(N_CLIENTS * 0.0986)   # ~9.9% churn rate (matches BCG dataset)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _random_dates(start: str, end: str, n: int) -> pd.Series:
    start_ts = pd.Timestamp(start).value
    end_ts = pd.Timestamp(end).value
    return pd.to_datetime(
        RNG.integers(start_ts, end_ts, n), unit="ns"
    ).normalize()


def generate_client_data() -> pd.DataFrame:
    ids = [f"client_{i:05d}" for i in range(N_CLIENTS)]
    churn = np.zeros(N_CLIENTS, dtype=int)
    churn[RNG.choice(N_CLIENTS, N_CHURNED, replace=False)] = 1

    channels = ["lev", "foo", "ewxo", "sddw", "ugen", "fdxy"]
    channel_probs = [0.30, 0.22, 0.18, 0.14, 0.09, 0.07]

    origins = ["kamkkxfh", "ldkssxwpm", "lmkwmag", "usapbeihkxs", "ewxeelcelemmkeh"]
    origin_probs = [0.28, 0.24, 0.20, 0.16, 0.12]

    n = N_CLIENTS
    date_activ = _random_dates("2010-01-01", "2018-01-01", n)
    date_end = date_activ + pd.to_timedelta(RNG.integers(365, 1825, n), unit="D")
    date_modif = date_activ + pd.to_timedelta(RNG.integers(0, 365, n), unit="D")
    date_renewal = date_end - pd.to_timedelta(RNG.integers(30, 90, n), unit="D")

    cons_12m = RNG.lognormal(mean=7.5, sigma=1.8, size=n).clip(0, 2_000_000).astype(int)
    cons_gas_12m = np.where(
        RNG.random(n) < 0.45,
        RNG.lognormal(mean=6.5, sigma=1.5, size=n).clip(0, 500_000).astype(int),
        0,
    )
    cons_last = (cons_12m / 12 * RNG.uniform(0.5, 1.8, n)).astype(int)
    has_gas = (cons_gas_12m > 0).astype(int)

    # Churners tend to have slightly higher consumption variance and lower margins
    margin_gross = RNG.normal(20, 15, n)
    margin_net = margin_gross - RNG.uniform(0, 5, n)
    margin_gross[churn == 1] *= RNG.uniform(0.6, 0.9, N_CHURNED)
    margin_net[churn == 1] *= RNG.uniform(0.6, 0.9, N_CHURNED)

    df = pd.DataFrame(
        {
            "id": ids,
            "activity_new": RNG.choice(
                ["", "esoiiifxo", "sddiedme", "lp", "dloehofbdk"], n,
                p=[0.60, 0.15, 0.12, 0.08, 0.05],
            ),
            "campaign_disc_ele": RNG.uniform(0, 0.3, n).round(4),
            "channel_sales": RNG.choice(channels, n, p=channel_probs),
            "cons_12m": cons_12m,
            "cons_gas_12m": cons_gas_12m,
            "cons_last_month": cons_last,
            "date_activ": date_activ.dt.strftime("%Y-%m-%d"),
            "date_end": date_end.dt.strftime("%Y-%m-%d"),
            "date_modif_prod": date_modif.dt.strftime("%Y-%m-%d"),
            "date_renewal": date_renewal.dt.strftime("%Y-%m-%d"),
            "forecast_cons_12m": (cons_12m * RNG.uniform(0.9, 1.1, n)).astype(int),
            "forecast_cons_year": (cons_12m * RNG.uniform(0.85, 1.15, n)).astype(int),
            "forecast_discount_energy": RNG.uniform(0, 0.25, n).round(4),
            "forecast_meter_rent_12m": RNG.uniform(0, 200, n).round(2),
            "forecast_price_energy_p1": RNG.uniform(0.05, 0.25, n).round(6),
            "forecast_price_energy_p2": RNG.uniform(0.03, 0.20, n).round(6),
            "forecast_price_pow_p1": RNG.uniform(20, 60, n).round(6),
            "has_gas": has_gas,
            "imp_cons": (cons_last * RNG.uniform(0.08, 0.20, n)).round(2),
            "margin_gross_pow_ele": margin_gross.round(2),
            "margin_net_pow_ele": margin_net.round(2),
            "nb_prod_act": RNG.choice([1, 2, 3, 4], n, p=[0.55, 0.25, 0.12, 0.08]),
            "net_margin": (margin_net * cons_12m / 10_000).round(2),
            "num_years_antig": RNG.integers(1, 15, n),
            "origin_up": RNG.choice(origins, n, p=origin_probs),
            "pow_max": RNG.choice(
                [3.45, 4.6, 5.75, 6.9, 10.35, 13.8, 17.25],
                n,
                p=[0.20, 0.18, 0.17, 0.17, 0.13, 0.10, 0.05],
            ),
            "churn": churn,
        }
    )

    # Inject ~3% nulls into a few columns to simulate real-world messiness
    for col in ["campaign_disc_ele", "activity_new", "channel_sales", "date_modif_prod"]:
        null_mask = RNG.random(n) < 0.03
        df.loc[null_mask, col] = np.nan

    return df


def generate_price_data(client_ids: list[str]) -> pd.DataFrame:
    """Monthly price records — 12 months per client."""
    months = pd.date_range("2015-01-01", periods=12, freq="MS")
    rows = []
    for cid in client_ids:
        base_off_peak_var = RNG.uniform(0.04, 0.12)
        base_peak_var = base_off_peak_var * RNG.uniform(1.1, 1.8)
        base_mid_peak_var = base_off_peak_var * RNG.uniform(1.0, 1.4)
        base_off_peak_fix = RNG.uniform(30, 70)
        base_peak_fix = base_off_peak_fix * RNG.uniform(0.8, 1.2)
        base_mid_peak_fix = base_off_peak_fix * RNG.uniform(0.9, 1.1)

        for m in months:
            noise = RNG.uniform(0.97, 1.03)
            rows.append(
                {
                    "id": cid,
                    "price_date": m.strftime("%Y-%m-%d"),
                    "price_off_peak_var": round(base_off_peak_var * noise, 6),
                    "price_peak_var": round(base_peak_var * noise, 6),
                    "price_mid_peak_var": round(base_mid_peak_var * noise, 6),
                    "price_off_peak_fix": round(base_off_peak_fix * noise, 6),
                    "price_peak_fix": round(base_peak_fix * noise, 6),
                    "price_mid_peak_fix": round(base_mid_peak_fix * noise, 6),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating client data...")
    client_df = generate_client_data()
    client_df.to_csv(OUTPUT_DIR / "client_data.csv", index=False)
    print(f"  Saved {len(client_df):,} client records  |  churn rate: {client_df['churn'].mean():.2%}")

    print("Generating price data...")
    price_df = generate_price_data(client_df["id"].tolist())
    price_df.to_csv(OUTPUT_DIR / "price_data.csv", index=False)
    print(f"  Saved {len(price_df):,} price records")

    print("Done. Files written to:", OUTPUT_DIR)
