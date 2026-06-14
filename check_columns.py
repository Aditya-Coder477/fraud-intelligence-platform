"""
Standalone utility: inspect Excel columns EV-EZ in DataSet_Engineered.csv,
report which are all-zero / constant, then write a cleaned CSV with:
  - All all-zero / constant duplicate columns dropped
  - One representative kept if all 5 are in the same constant group

Run from the project directory with:
    python column_inspector.py
"""
import sys
import os
import pandas as pd
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────
INPUT_CSV  = "DataSet_Engineered.csv"
OUTPUT_CSV = "DataSet_Engineered_Cleaned.csv"
TARGET     = "F3924"

# Excel columns the user flagged
TARGET_EXCEL_COLS = ["EV", "EW", "EX", "EY", "EZ"]

# ─── Helper: Excel column letter → 0-based integer index ──────────────────
def excel_col_to_idx(letters: str) -> int:
    """'A'→0, 'Z'→25, 'AA'→26, 'EV'→151, etc."""
    result = 0
    for ch in letters.upper():
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1  # 0-based

# ─── Helper: comprehensive constant/near-constant column detector ──────────
def find_useless_cols(df: pd.DataFrame, target_col: str) -> dict:
    """
    Returns dict with keys:
        all_zero     : columns where every non-NaN value is 0 (and NaN filled = 0)
        constant     : columns with ≤ 1 unique non-NaN value
        near_constant: columns where top value accounts for > 99.9 % of rows
    """
    result = {"all_zero": [], "constant": [], "near_constant": []}
    for c in df.columns:
        if c == target_col:
            continue
        series = df[c]
        filled = series.fillna(0)

        # All-zero check (including NaN→0)
        if (filled == 0).all():
            result["all_zero"].append(c)
            continue

        # Constant (≤1 unique non-NaN value)
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            result["constant"].append(c)
            continue

        # Near-constant (>99.9 % dominated by a single value)
        top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.999:
            result["near_constant"].append(c)

    return result


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌  '{INPUT_CSV}' not found. Run feature_engineering_pipeline.py first.")
        sys.exit(1)

    # ── Step 1: Read header only, resolve Excel column letters ─────────────
    print(f"Reading header of '{INPUT_CSV}' …")
    header = pd.read_csv(INPUT_CSV, nrows=0)
    all_cols = header.columns.tolist()
    total_cols = len(all_cols)
    print(f"  Total columns : {total_cols}")

    print("\n── Resolving Excel column positions ─────────────────────────────────")
    flagged_name_to_letter = {}   # col_name → excel letter
    out_of_range = []
    for letter in TARGET_EXCEL_COLS:
        idx = excel_col_to_idx(letter)
        if idx < total_cols:
            col_name = all_cols[idx]
            flagged_name_to_letter[col_name] = letter
            print(f"  Column {letter:>3s}  (index {idx:>4d})  →  '{col_name}'")
        else:
            out_of_range.append(letter)
            print(f"  Column {letter:>3s}  (index {idx:>4d})  →  OUT OF RANGE")

    if out_of_range:
        print(f"\n  ⚠  {out_of_range} are out of range — only {total_cols} columns exist.")

    # ── Step 2: Load full CSV and run diagnostics ──────────────────────────
    print(f"\nLoading full '{INPUT_CSV}' …")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Detailed report on just the flagged columns
    flagged_cols = list(flagged_name_to_letter.keys())
    print("\n── Detailed inspection of flagged columns ───────────────────────────")
    print(f"{'Letter':>6}  {'Column Name':<55} {'All-Zero':>9} {'Unique':>7} {'NaN%':>7} {'Min':>10} {'Max':>10}")
    print("─" * 110)

    confirmed_all_zero = []
    for col, letter in flagged_name_to_letter.items():
        if col not in df.columns:
            print(f"  {letter:>4}  {col:<55}  NOT IN DATAFRAME")
            continue
        s          = df[col]
        filled     = s.fillna(0)
        is_allzero = (filled == 0).all()
        n_unique   = s.nunique(dropna=False)
        nan_pct    = s.isna().mean() * 100
        col_min    = s.min() if not s.isna().all() else float("nan")
        col_max    = s.max() if not s.isna().all() else float("nan")

        flag = "✅ YES" if is_allzero else "❌  NO"
        print(f"  {letter:>4}  {col:<55} {flag:>9}  {n_unique:>7}  {nan_pct:>6.2f}%  {col_min:>10.4f}  {col_max:>10.4f}")

        if is_allzero:
            confirmed_all_zero.append(col)

    print("─" * 110)

    # ── Step 3: Global scan for ALL useless columns ────────────────────────
    print("\nScanning ALL columns for zero / constant / near-constant …")
    useless = find_useless_cols(df, TARGET)

    print(f"  All-zero columns      : {len(useless['all_zero'])}")
    print(f"  Constant columns      : {len(useless['constant'])}")
    print(f"  Near-constant (>99.9%): {len(useless['near_constant'])}")

    all_to_drop = list(set(useless["all_zero"] + useless["constant"] + useless["near_constant"]))
    # Never drop the target
    all_to_drop = [c for c in all_to_drop if c != TARGET]

    # ── Step 4: Decision on the 5 flagged columns ─────────────────────────
    print("\n" + "═" * 70)
    print("  DECISION — flagged columns EV / EW / EX / EY / EZ")
    print("═" * 70)

    if len(confirmed_all_zero) == len(flagged_cols) and len(flagged_cols) > 0:
        print(f"\n  All {len(flagged_cols)} flagged columns are identically zero.")
        print("  These carry ZERO information for the model.")
        print("  → ACTION: Drop all 5 and replace with a SINGLE indicator column")
        print("    named 'zero_group_EV_EZ_placeholder' = 0 (constant, will also")
        print("    be dropped in the final clean step).")
        print("  → FINAL ACTION: Drop all 5 entirely — no placeholder needed.")
    elif len(confirmed_all_zero) > 0:
        print(f"\n  {len(confirmed_all_zero)} of {len(flagged_cols)} flagged columns are all-zero:")
        for c in confirmed_all_zero:
            print(f"    → Drop: {c}")
        non_zero = [c for c in flagged_cols if c not in confirmed_all_zero]
        print(f"  {len(non_zero)} have meaningful values — KEEP:")
        for c in non_zero:
            print(f"    → Keep: {c}")
    else:
        print(f"\n  None of the flagged columns are all-zero.")
        print("  No action required based on your observation.")
        print("  (They may look like zero in Excel due to very small float values.)")

    # ── Step 5: Write cleaned CSV ─────────────────────────────────────────
    cols_before = df.shape[1]
    df_clean = df.drop(columns=all_to_drop, errors="ignore")
    cols_after = df_clean.shape[1]
    dropped = cols_before - cols_after

    print(f"\n── Global Cleanup Summary ───────────────────────────────────────────")
    print(f"  Columns before cleanup : {cols_before}")
    print(f"  Columns dropped        : {dropped}  (zero + constant + near-constant)")
    print(f"  Columns after cleanup  : {cols_after}")
    print(f"  Target column present  : {'YES' if TARGET in df_clean.columns else 'NO'}")

    # Show which of the 5 flagged were dropped
    dropped_flagged = [c for c in flagged_cols if c in all_to_drop]
    kept_flagged    = [c for c in flagged_cols if c not in all_to_drop]
    print(f"\n  Of the 5 flagged columns:")
    print(f"    Dropped : {len(dropped_flagged)}  →  {dropped_flagged}")
    print(f"    Kept    : {len(kept_flagged)}   →  {kept_flagged}")

    # Save
    print(f"\nSaving cleaned dataset → '{OUTPUT_CSV}' …")
    df_clean.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1_048_576
    print(f"  ✅  Saved  ({size_mb:.1f} MB)  —  shape: {df_clean.shape[0]:,} × {df_clean.shape[1]:,}")

    # ── Step 6: Print dropped column list ─────────────────────────────────
    print(f"\n── All {dropped} dropped columns ───────────────────────────────────────")
    for category, cols in useless.items():
        actual_dropped = [c for c in cols if c in all_to_drop]
        if actual_dropped:
            print(f"\n  [{category.upper()}]  ({len(actual_dropped)} columns)")
            for c in actual_dropped[:20]:   # show first 20
                print(f"    {c}")
            if len(actual_dropped) > 20:
                print(f"    … and {len(actual_dropped) - 20} more")

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
