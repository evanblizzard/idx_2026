import pandas as pd
import numpy as np
import os

# -----------------------------------------------
# Week 6: Feature Engineering and Market Metrics
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
sold = pd.read_csv(os.path.join(DATA_DIR, 'sold_cleaned.csv'))
listings = pd.read_csv(os.path.join(DATA_DIR, 'listings_cleaned.csv'))

print("=" * 70)
print("Week 6: Feature Engineering and Market Metrics")
print("=" * 70)
print(f"\nStarting row counts:")
print(f"   sold: {len(sold):,} rows")
print(f"   listings: {len(listings):,} rows")


def add_market_metrics(df, label):
    df = df.copy()

    for col in ['CloseDate', 'PurchaseContractDate', 'ListingContractDate']:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    # Price Ratio: how close final sale price landed to original ask
    if {'ClosePrice', 'OriginalListPrice'}.issubset(df.columns):
        df['price_ratio'] = df['ClosePrice'] / df['OriginalListPrice']

        df['close_to_original_list_ratio'] = df['ClosePrice'] / df['OriginalListPrice']

    # price per sq ft: normalizes price across different home sizes
    if {'ClosePrice', 'LivingArea'}.issubset(df.columns):
        # guard against divide by zero on LivingArea
        df['price_per_sqft'] = np.where(
            df['LivingArea'] > 0,
            df['ClosePrice'] / df['LivingArea'],
            np.nan
        )

    # Year / month / YrMo: derived from CloseDate for time-series grouping
    if 'CloseDate' in df.columns:
        df['sale_year'] = df['CloseDate'].dt.year
        df['sale_month'] = df['CloseDate'].dt.month
        df['sale_yrmo'] = df['CloseDate'].dt.to_period('M').astype(str)

    # Listing to Contract Days: how long a home sat before going under contract
    if {'PurchaseContractDate', 'ListingContractDate'}.issubset(df.columns):
        df['listing_to_contract_days'] = (
            df['PurchaseContractDate'] - df['ListingContractDate']
        ).dt.days

    # Contract to Close Days: escrow/closing duration
    if {'CloseDate', 'PurchaseContractDate'}.issubset(df.columns):
        df['contract_to_close_days'] = (
            df['CloseDate'] - df['PurchaseContractDate']
        ).dt.days

    new_cols = [c for c in [
        'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
        'sale_year', 'sale_month', 'sale_yrmo',
        'listing_to_contract_days', 'contract_to_close_days'
    ] if c in df.columns]

    print(f"\n[{label}] Added metrics: {new_cols}")
    return df
sold = add_market_metrics(sold, "sold")
listings = add_market_metrics(listings, "listings")

sample_cols = [c for c in [
    'ClosePrice', 'OriginalListPrice', 'ListPrice', 'LivingArea',
    'price_ratio', 'close_to_original_list_ratio', 'price_per_sqft',
    'sale_yrmo', 'listing_to_contract_days', 'contract_to_close_days',
    'DistrictName'
] if c in sold.columns]
 
print("\n" + "=" * 70)
print("SAMPLE OUTPUT (sold, 5 rows)")
print("=" * 70)
print(sold[sample_cols].head())



# -------------------------------------------------------------
# Segment analysis
# -------------------------------------------------------------
def segment_summary(df, group_col, label):
    if group_col not in df.columns or 'ClosePrice' not in df.columns:
        print(f"\nSkipping segment summary by {group_col} -- column(s) missing")
        return None
 
    summary = df.groupby(group_col).agg(
        transaction_count=('ClosePrice', 'count'),
        median_close_price=('ClosePrice', 'median'),
        median_dom=('DaysOnMarket', 'median') if 'DaysOnMarket' in df.columns else ('ClosePrice', 'count'),
        median_price_per_sqft=('price_per_sqft', 'median') if 'price_per_sqft' in df.columns else ('ClosePrice', 'count'),
    ).sort_values('transaction_count', ascending=False)
 
    print(f"\n[{label}] Segment summary by {group_col} (top 10 by volume):")
    print(summary.head(10))
    return summary
 
 
by_property_type = segment_summary(sold, 'PropertyType', 'sold')
by_county = segment_summary(sold, 'CountyOrParish', 'sold')

sold.to_csv(os.path.join(DATA_DIR, 'sold_engineered.csv'), index=False)
listings.to_csv(os.path.join(DATA_DIR, 'listings_engineered.csv'), index=False)

print(f"\nSaved engineered datasets:")
print(f"   {os.path.join(DATA_DIR, 'sold_engineered.csv')}")
print(f"   {os.path.join(DATA_DIR, 'listings_engineered.csv')}")
