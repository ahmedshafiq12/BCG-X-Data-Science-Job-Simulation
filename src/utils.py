"""Shared plotting and preprocessing helpers used across notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    ax = dataframe.plot(kind="bar", stacked=True, figsize=size_, rot=rot_, title=title_)
    _annotate_stacked_bars(ax, textsize=14)
    plt.legend(["Retention", "Churn"], loc=legend_)
    plt.ylabel("Company base (%)")
    plt.show()


def _annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    for p in ax.patches:
        value = str(round(p.get_height(), 1))
        if value == "0.0":
            continue
        ax.annotate(
            value,
            ((p.get_x() + p.get_width() / 2) * pad - 0.05,
             (p.get_y() + p.get_height() / 2) * pad),
            color=colour,
            size=textsize,
        )


def plot_distribution(dataframe, column, ax, bins_=50):
    temp = pd.DataFrame({
        "Retention": dataframe[dataframe["churn"] == 0][column],
        "Churn":     dataframe[dataframe["churn"] == 1][column],
    })
    temp[["Retention", "Churn"]].plot(kind="hist", bins=bins_, ax=ax, stacked=True)
    ax.set_xlabel(column)
    ax.ticklabel_format(style="plain", axis="x")


def missing_summary(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    result = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    result = result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    label = f" ({name})" if name else ""
    print(f"Missing values{label}:")
    print(result.to_string() if not result.empty else "  None")
    return result


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def clean_client_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard cleaning pipeline to client_data."""
    df = df.copy()
    df["has_gas"] = df["has_gas"].map({"t": 1, "f": 0})
    for col in ["date_activ", "date_end", "date_modif_prod", "date_renewal"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["channel_sales"] = df["channel_sales"].replace("MISSING", np.nan)
    cap = df["nb_prod_act"].quantile(0.99)
    df["nb_prod_act"] = df["nb_prod_act"].clip(upper=cap)
    return df


def engineer_temporal_features(df: pd.DataFrame, ref_date: str = "2016-01-01") -> pd.DataFrame:
    ref = pd.Timestamp(ref_date)
    df["tenure_months"]          = ((ref - df["date_activ"]).dt.days / 30.44).round(1)
    df["contract_duration_days"] = (df["date_end"] - df["date_activ"]).dt.days
    df["months_to_end"]          = ((df["date_end"] - ref).dt.days / 30.44).clip(lower=0).round(1)
    df["months_since_modif"]     = ((ref - df["date_modif_prod"]).dt.days / 30.44).round(1)
    return df


def engineer_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    agg = price_df.groupby("id").agg(
        mean_off_peak_var = ("price_off_peak_var", "mean"),
        mean_peak_var     = ("price_peak_var",      "mean"),
        mean_off_peak_fix = ("price_off_peak_fix",  "mean"),
        mean_peak_fix     = ("price_peak_fix",       "mean"),
        std_off_peak_var  = ("price_off_peak_var",  "std"),
        std_peak_var      = ("price_peak_var",       "std"),
    ).reset_index().fillna(0)

    agg["peak_offpeak_spread_var"] = agg["mean_peak_var"] - agg["mean_off_peak_var"]
    agg["peak_offpeak_spread_fix"] = agg["mean_peak_fix"] - agg["mean_off_peak_fix"]
    return agg
