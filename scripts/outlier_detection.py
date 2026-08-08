import pandas as pd
import os

# setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')

INPUT_PATH = os.path.join(DATA_DIR, "sold_engineered.csv")
FLAGGED_OUTPUT_PATH = os.path.join(DATA_DIR, "sold_flagged_full.csv")
CLEAN_OUTPUT_PATH = os.path.join(DATA_DIR, "sold_filtered_clean.csv")

# fields specified for IQR filtering
FIELDS_TO_CHECK = ['ClosePrice', 'LivingArea', 'DaysOnMarket']

# IQR Multiplier
IQR_MULTIPLIER = 1.5

def compute_iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Return (lower_bound, upper_bound) for a numeric series using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper

def flag_outliers(df: pd.DataFrame, fields: list[str], multiplier: float = 1.5) -> pd.DataFrame: 
    """
    Adds one boolean flag column per field (e.g. 'ClosePrice_outlier') and a
    combined 'is_outlier' column (True if ANY field is flagged for that row).
    Does not drop or modify any existing data -- purely additive.
    """
    df = df.copy()
    combined_flag = pd.Series(False, index=df.index)
 
    print("IQR bounds used for flagging:")
    for field in fields:
        if field not in df.columns:
            print(f"  [skip] '{field}' not found in dataset")
            continue
 
        lower, upper = compute_iqr_bounds(df[field], multiplier)
        flag_col = f"{field}_outlier"
        df[flag_col] = (df[field] < lower) | (df[field] > upper)
 
        combined_flag = combined_flag | df[flag_col]
 
        print(f"  {field}: lower={lower:,.2f}, upper={upper:,.2f}, "
              f"flagged={df[flag_col].sum():,} rows")
 
    df["is_outlier"] = combined_flag
    return df

def print_before_after_summary(df_full: pd.DataFrame, df_clean: pd.DataFrame, fields: list[str]) -> None:
    """Prints row counts and median values before/after filtering for each field."""
    print("\n" + "=" * 60)
    print("BEFORE / AFTER COMPARISON")
    print("=" * 60)
    print(f"Row count before filtering: {len(df_full):,}")
    print(f"Row count after filtering:  {len(df_clean):,}")
    print(f"Rows removed:               {len(df_full) - len(df_clean):,} "
          f"({(len(df_full) - len(df_clean)) / len(df_full):.1%})")
 
    print("\nMedian values (before -> after):")
    for field in fields:
        if field not in df_full.columns:
            continue
        med_before = df_full[field].median()
        med_after = df_clean[field].median()
        print(f"  {field}: {med_before:,.2f}  ->  {med_after:,.2f}")


def main() -> None:
    # 1. Load Week 6 engineered dataset
    df = pd.read_csv(INPUT_PATH)
    print(f"Total rows: {len(df):,}")
    print(df[['ClosePrice', 'LivingArea', 'DaysOnMarket']].describe())
    print(f"Loaded {INPUT_PATH}: {len(df):,} rows, {len(df.columns)} columns")
 
    # 2. Apply business-rule validity checks first (always invalid, per handbook)
    #     separate from statistical outliers -- e.g. ClosePrice <= 0
    #    should already have been handled in Weeks 4-5, but guard here too.
    valid_mask = (
        (df.get("ClosePrice", pd.Series(1, index=df.index)) > 0)
        & (df.get("LivingArea", pd.Series(1, index=df.index)) > 0)
        & (df.get("DaysOnMarket", pd.Series(0, index=df.index)) >= 0)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"\nWarning: {invalid_count:,} rows fail basic business-rule "
              f"validity checks (ClosePrice<=0, LivingArea<=0, DaysOnMarket<0). "
              f"These should have been resolved in Weeks 4-5.")
 
    # 3. Flag statistical outliers via IQR (additive, non-destructive)
    df_flagged = flag_outliers(df, FIELDS_TO_CHECK, IQR_MULTIPLIER)
 
    # 4. Build the filtered/clean dataset (outlier rows AND invalid rows removed)
    df_clean = df_flagged[(~df_flagged["is_outlier"]) & valid_mask].copy()
 
    # 5. Save both outputs
    df_flagged.to_csv(FLAGGED_OUTPUT_PATH, index=False)
    df_clean.to_csv(CLEAN_OUTPUT_PATH, index=False)
    print(f"\nSaved full flagged dataset  -> {FLAGGED_OUTPUT_PATH} ({len(df_flagged):,} rows)")
    print(f"Saved filtered clean dataset -> {CLEAN_OUTPUT_PATH} ({len(df_clean):,} rows)")
 
    # 6. Written before/after comparison (required in Week 7 deliverable)
    print_before_after_summary(df_flagged, df_clean, FIELDS_TO_CHECK)
 
 
if __name__ == "__main__":
    main()
