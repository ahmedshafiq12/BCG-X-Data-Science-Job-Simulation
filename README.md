# BCG X Data Science Job Simulation — PowerCo Churn Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Forage](https://img.shields.io/badge/BCG%20X-Data%20Science%20Simulation-blue)](https://www.theforage.com/virtual-experience/Tcz8gTtprzAS4xSoK/bcg)

## Overview

This repository contains my end-to-end solution to the **BCG X Data Science Virtual Experience** on [Forage](https://www.theforage.com/virtual-experience/Tcz8gTtprzAS4xSoK/bcg). The simulation mirrors real client work at BCG X, where data scientists tackle strategic business problems using machine learning.

**Client:** PowerCo — a major European gas & electricity utility supplying SME customers  
**Business problem:** Significant customer churn threatening revenue; PowerCo wants to know *why* customers leave and *who* is most at risk

---

## Project Structure

```
BCG-X-Data-Science/
├── data/
│   ├── raw/               ← Original Forage datasets (not committed — see below)
│   └── processed/         ← Cleaned & feature-engineered outputs
├── models/                ← Trained Random Forest model (.pkl)
├── notebooks/
│   ├── 01_business_understanding.ipynb     ← Task 1: Hypothesis & SME questions
│   ├── 02_eda_data_cleaning.ipynb          ← Task 2: EDA & cleaning
│   ├── 03_feature_engineering_modelling.ipynb  ← Task 3: Feature eng. & RF model
│   └── 04_findings_recommendations.ipynb   ← Task 4: Insights & business case
├── reports/figures/       ← All exported charts
├── src/
│   └── utils.py           ← Shared plotting & preprocessing functions
├── requirements.txt
└── README.md
```

---

## Tasks & Key Work

### Task 1 — Business Understanding & Hypothesis Framing
- Defined the analytical problem as a **binary churn classification** task
- Framed the central hypothesis: *is churn driven primarily by price sensitivity?*
- Drafted 11 SME questions to align on business context and data definitions
- Mapped available data to testable sub-hypotheses

### Task 2 — Exploratory Data Analysis & Data Cleaning
- Profiled 14,606 SME customers and 193,002 monthly price records
- Identified and fixed: string booleans (`has_gas`), object-typed dates, `'MISSING'` sentinel values, and outlier `nb_prod_act` entries
- Visualised churn distributions by sales channel, origin campaign, tenure, and gas subscription
- Analysed price trends and compared pricing profiles of churned vs retained customers
- **Finding:** 9.7% overall churn rate; tenure and margins are stronger signals than price alone

### Task 3 — Feature Engineering & Modelling
Engineered 3 groups of features on top of the raw data:

| Group | Features |
|---|---|
| Price features | Mean off-peak/peak prices, peak–offpeak spread, price volatility (std), Jan→Dec price deltas |
| Temporal features | Tenure in months, contract duration, months to contract end, months since last modification |
| Consumption ratios | Last-month vs annual average, gas share of total consumption, log transforms |

**Model:** Random Forest Classifier (500 trees, `class_weight='balanced'`)  
**Performance:**
- ROC-AUC (hold-out): **≥ 0.73**
- 5-fold CV ROC-AUC: stable across folds

### Task 4 — Findings & Recommendations
- Partially disconfirmed the pure price-sensitivity hypothesis — **financial margin and tenure dominate**
- Built a customer risk segmentation (High / Medium / Low / Very Low risk tiers)
- Modelled the **business case for a 20% discount campaign** with sensitivity analysis
- Delivered 4 actionable recommendations: targeted discounts, gas cross-sell, new-customer onboarding, and quarterly retraining

---

## Key Findings

1. **Churn is not primarily price-driven** — net margin and customer tenure are stronger predictors than energy prices
2. **Newer customers (1–3 years)** churn at the highest rate — early-tenure loyalty investment is critical
3. **Dual-fuel (gas + electricity) customers** churn significantly less — cross-selling gas is the most powerful structural retention lever
4. **A 0.30 probability threshold** optimally balances precision and recall for campaign targeting
5. The 20% discount campaign is **ROI-positive** if retention uplift exceeds ~10–15%

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/bcg-x-powerco-churn.git
cd bcg-x-powerco-churn

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

> **Data:** The raw CSVs are available by completing the [BCG X Forage simulation](https://www.theforage.com/virtual-experience/Tcz8gTtprzAS4xSoK/bcg). Place them in `data/raw/` as `client_data.csv` and `price_data.csv`.

---

## Full Report

A detailed markdown report covering all four tasks — EDA findings, model evaluation, risk segmentation, business case, and recommendations — is available at:

**[reports/REPORT.md](reports/REPORT.md)**

---

## Results Snapshot

| Notebook | Key output |
|---|---|
| `01_business_understanding` | 7 testable sub-hypotheses, 11 SME questions |
| `02_eda_data_cleaning` | 19 EDA visualisations, cleaned datasets in `data/processed/` |
| `03_feature_engineering_modelling` | 40+ features, ROC-AUC ≥ 0.73, threshold analysis |
| `04_findings_recommendations` | Risk segmentation, discount ROI model, action matrix |

---

## Technologies

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Modelling, evaluation |
| `joblib` | Model serialisation |

---

*Completed as part of the BCG X Data Science Virtual Experience Programme on Forage.*
