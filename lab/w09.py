"""EEE213 Week 09 lab: machine learning on the power-plant (CCPP) data.

Run from the repository folder:
    python lab/w09.py

Data: data/ccpp.csv (columns AT [degC], V [cm Hg], AP [mbar], RH [%], PE [MW]).
Training set: first 7,654 rows. Test set: last 1,914 rows.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "ccpp.csv"
N_TRAIN = 7654
N_TEST = 1914
UNITS = {"AT": "degC", "V": "cm Hg", "AP": "mbar", "RH": "%", "PE": "MW"}
FEATURE = "V"          # feature for the straight line in 5.2 (not AT)
CHECK_INDEX = 0        # which test row to check by hand in 5.5 (0 = first test row)


def mae(y_true, y_pred):
    """Mean absolute error, in the unit of y."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def fit_line(x, y):
    """Least-squares line y = a*x + b. Returns (a, b)."""
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def main():
    df = pd.read_csv(DATA)

    # ---------------- Section 3: data-quality checks ----------------
    print("=== Section 3: data quality ===")
    print(f"Rows: {len(df)}")
    print("Columns: " + ", ".join(f"{c} [{UNITS[c]}]" for c in df.columns))
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print("Missing values per column: " + ", ".join(f"{c}={int(n)}" for c, n in df.isna().sum().items()))
    n_dup = int(df.duplicated().sum())
    print(f"Exact duplicate rows (repeats of an earlier row): {n_dup}")
    print("Min / max per column:")
    for c in df.columns:
        print(f"  {c}: {df[c].min():.2f} - {df[c].max():.2f} {UNITS[c]}")

    if len(df) != N_TRAIN + N_TEST:
        sys.exit(f"Expected {N_TRAIN + N_TEST} rows, found {len(df)}")
    train = df.iloc[:N_TRAIN]
    test = df.iloc[N_TRAIN:]
    print(f"Training rows: {len(train)}, test rows: {len(test)} (all rows kept)")
    print(f"Duplicates inside training set: {int(train.duplicated().sum())}, "
          f"inside test set: {int(test.duplicated().sum())}, "
          f"test rows that also appear in training set: "
          f"{int(test.apply(tuple, axis=1).isin(set(train.apply(tuple, axis=1))).sum())}")

    # ---------------- Section 4: summary statistics and correlation ----------------
    print("\n=== Section 4: summary statistics (full dataset) ===")
    for c in ["AT", "PE"]:
        print(f"{c}: mean {df[c].mean():.2f}, median {df[c].median():.2f}, "
              f"std {df[c].std():.2f} {UNITS[c]}")
    print("Correlation r with PE:")
    r = df.corr()["PE"].drop("PE")
    for c, val in r.items():
        print(f"  {c}: {val:.3f}")
    strongest = r.abs().idxmax()
    print(f"Strongest linear relationship: {strongest} (r = {r[strongest]:.3f})")

    # ---------------- Section 5: models and test error ----------------
    print("\n=== Section 5: models (fit on training set, evaluate on test set) ===")
    y_train, y_test = train["PE"], test["PE"]

    mean_pe = float(y_train.mean())
    mae_base = mae(y_test, np.full(len(y_test), mean_pe))
    print(f"5.1 Training mean of PE: {mean_pe:.2f} MW")
    print(f"5.1 Baseline test MAE: {mae_base:.2f} MW")

    a_at, b_at = fit_line(train["AT"], y_train)
    mae_at = mae(y_test, a_at * test["AT"] + b_at)
    print(f"Check - line from AT: PE = {a_at:.4f}*AT + {b_at:.2f}, test MAE {mae_at:.2f} MW")

    a, b = fit_line(train[FEATURE], y_train)
    mae_line = mae(y_test, a * test[FEATURE] + b)
    print(f"5.2 Feature: {FEATURE} [{UNITS[FEATURE]}]")
    print(f"5.2 Slope a: {a:.4f} MW per {UNITS[FEATURE]}")
    print(f"5.2 Intercept b: {b:.2f} MW")
    print(f"5.2 Test MAE: {mae_line:.2f} MW")

    print("5.3 Comparison table (test MAE, MW):")
    print(f"  Baseline (training mean): {mae_base:.2f}")
    print(f"  Line from AT:             {mae_at:.2f}")
    print(f"  Line from {FEATURE}:              {mae_line:.2f}")

    # 5.5: one test row checked by hand. File line = data index + 2 (header is line 1).
    idx = N_TRAIN + CHECK_INDEX
    x = float(df.loc[idx, FEATURE])
    true_pe = float(df.loc[idx, "PE"])
    pred = a * x + b
    print(f"5.5 Row number in file (header is line 1): {idx + 2}")
    print(f"5.5 {FEATURE} = {x} {UNITS[FEATURE]}")
    print(f"5.5 Predicted PE = {a:.4f}*{x} + {b:.2f} = {pred:.2f} MW")
    print(f"5.5 True PE: {true_pe:.2f} MW")
    print(f"5.5 Absolute error: {abs(pred - true_pe):.2f} MW")

    # Separate experiment for 3.3: remove duplicates, same split rule applied per set.
    train_d = train.drop_duplicates()
    test_d = test.drop_duplicates()
    a_d, b_d = fit_line(train_d[FEATURE], train_d["PE"])
    print("\n=== Extra (3.3): duplicates removed within each set ===")
    print(f"Training rows: {len(train_d)}, test rows: {len(test_d)}")
    print(f"Baseline test MAE: {mae(test_d['PE'], np.full(len(test_d), train_d['PE'].mean())):.2f} MW")
    print(f"Line from {FEATURE} test MAE: {mae(test_d['PE'], a_d * test_d[FEATURE] + b_d):.2f} MW")

    print(f"\nPython {sys.version.split()[0]}, NumPy {np.__version__}, pandas {pd.__version__}")


if __name__ == "__main__":
    main()
