import pandas as pd
import numpy as np 
import os
# -------------------------------------------
# Loading Data
# -------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
sold = pd.read_csv(os.path.join(DATA_DIR, 'sold_with_rates.csv'))
listings = pd.read_csv(os.path.join(DATA_DIR, 'listings_with_rates.csv'))

print("=" * 70)
print("Weeks 4-5: Data Cleaning and Preparation")
print("=" * 70)
print(f"\nStarting row counts:")
print(f"   sold: {len(sold):,} rows, {sold.shape[1]} columns")
print(f"   listings: {len(listings):,} rows, {listings.shape[1]} columns")


# -------------------------------------------------------------
# 1. Resolving duplicate '.1' columns from Week 1 concatenation
# -------------------------------------------------------------
# When sold/listings shared column names during merges and concats, pandas
# auto-suffixed duplicates with '.1'
# Inspecting each pair and keeping whichever version has fewer nulls, then dropping the other

def resolve_dot1_duplicates(df, label):
    dot1_cols = [c for c in df.columns if c.endswith('.1')]
    resolved_log = []


    for dup_col in dot1_cols:
        base_col = dup_col[:-2] # strip '.1'
        if base_col not in df.columns:
            continue

        base_nulls = df[base_col].isnull().sum()
        dup_nulls = df[dup_col].isnull().sum()

        # check if they're really identical
        identical = df[base_col].equals(df[dup_col])

        if identical:
            df.drop(columns=[dup_col], inplace=True)
            resolved_log.append(f"  {dup_col}: identical to {base_col} -> dropped {dup_col}")
        
        elif dup_nulls < base_nulls:
            # dup_col is more complete -> keep it, rename over base_col
            df.drop(columns=[base_col], inplace=True)
            df.rename(columns={dup_col: base_col}, inplace=True)
            resolved_log.append(
                f" {dup_col}: more complete ({dup_nulls} vs {base_nulls} nulls) "
                f"-> kept {dup_col}, dropped original {base_col}"
            )
        else:
            df.drop(columns=[dup_col], inplace=True)
            resolved_log.append(
                f" {dup_col}: original {base_col} more/equally complete "
                f"({base_nulls} vs {dup_nulls} nulls) -> dropped {dup_col}"
            )

    print(f"\n[{label}] Resolved {len(dot1_cols)} duplicate '.1' column(s)")
    for line in resolved_log:
        print(line)
    if not dot1_cols:
        print("    (none found)")
    

    return df

sold = resolve_dot1_duplicates(sold, "sold")
listings = resolve_dot1_duplicates(listings, "listings")

# --------------------------------------
# Convert Date Fields to Datetime
# --------------------------------------

date_cols_sold = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']
date_cols_listings = ['ListingContractDate', 'ContractStatusChangeDate']

for col in date_cols_sold:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors='coerce')
for col in date_cols_listings:
    if col in listings.columns:
        listings[col] = pd.to_datetime(listings[col], errors='coerce')

print("\nDate Fields converted to datetime:")
print(f"   sold: {[c for c in date_cols_sold if c in sold.columns]}")
print(f"   listings: {[c for c in date_cols_listings if c in listings.columns]}")



# ---------------------------------------
# Removing unnecessary/redundant columns
# ---------------------------------------

cols_to_drop_common = [
    # 100% empty
    'TaxYear', 'TaxAnnualAmount', 'AboveGradeFinishedArea', 'FireplacesTotal',
    'ElementarySchoolDistrict', 'CoveredSpaces', 'BusinessType', 'MiddleOrJuniorSchoolDistrict',
    # near-total missing (>90% and not needed for anything in handbook)
    'BelowGradeFinishedArea', 'BuilderName', 'LotSizeDimensions', 'BuildingAreaTotal', 'CoBuyerAgentFirstName',
    'ElementarySchool', 'MiddleOrJuniorSchool', 'HighSchool',
    'BuyerAgencyCompensation', 'BuyerAgencyCompensationType',
    # co-listing fields not required
    'CoListAgentFirstName', 'CoListAgentLastName', 'CoListOfficeName'
]
cols_to_drop_sold = cols_to_drop_common + [
    # near-total missing in sold
    'WaterfrontYN', "BasementYN",
    # pipeline metadata, not transaction data
    'OriginatingSystemName', 'OriginatingSystemSubName',
]

cols_to_drop_listings = cols_to_drop_common + [
    # none that I can find
]

before_cols_sold = sold.shape[1]
before_cols_listings = listings.shape[1]

sold.drop(columns=[c for c in cols_to_drop_sold if c in sold.columns], inplace=True)
listings.drop(columns=[c for c in cols_to_drop_listings if c in listings.columns], inplace=True)

print(f"\nColumn Cleanup:")
print(f"    sold: {before_cols_sold} -? {sold.shape[1]} columns")
print(f"    listings: {before_cols_listings} -> {listings.shape[1]} columns")



# ---------------------------------------------
# Ensuring numeric fields are properly typed
# ---------------------------------------------

numeric_cols = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres', 'BedroomsTotal',
                'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt']
for col in numeric_cols:
    if col in sold.columns:
        sold[col] = pd.to_numeric(sold[col], errors='coerce')
    if col in listings.columns:
        listings[col] = pd.to_numeric(listings[col], errors='coerce')

print(f"\nNumeric fields validated/coerced: {numeric_cols}")


# -------------------------------------------------------------
# Flagging invalid numeric values (not deleting, just flagging)
# -------------------------------------------------------------

def add_invalid_numeric_flags(df):
    df['invalid_price_flag'] = df['ClosePrice'] <=0 if 'ClosePrice' in df.columns else False
    df['invalid_livingarea_flag'] = df['LivingArea'] <= 0 if 'LivingArea' in df.columns else False
    df['invalid_dom_flag'] = df['DaysOnMarket'] < 0 if 'DaysOnMarket' in df.columns else False
    df['invalid_bedrooms_flag'] = df['BedroomsTotal'] < 0 if 'BedroomsTotal' in df.columns else False
    df['invalid_bathrooms_flag'] = df['BathroomsTotalInteger'] < 0 if 'BathroomsTotalInteger' in df.columns else False
    return df

sold = add_invalid_numeric_flags(sold)

invalid_flag_cols = ['invalid_price_flag', 'invalid_livingarea_flag', 'invalid_dom_flag', 'invalid_bedrooms_flag',
                     'invalid_bathrooms_flag']

print("\nInvalid numeric value flags (sold dataset):")
for flag in invalid_flag_cols:
    print(f" {flag}: {sold[flag].sum():,} flagged rows")

# --------------------------------
# Date Consistency Checks
# -------------------------------

# expected order: ListingContractDate <= PurchaseContractDate <= CloseDate
sold['listing_after_close_flag'] = sold['ListingContractDate'] > sold['CloseDate']
sold['purchase_after_close_flag'] = sold['PurchaseContractDate'] > sold['CloseDate']
sold['negative_timeline_flag'] = (
    sold['listing_after_close_flag'] | sold['purchase_after_close_flag']
)

print("\nDate consistency flags:")
print(f"   listing_after_close_flag: {sold['listing_after_close_flag'].sum():,} rows")
print(f"   purchase_after_close_flag: {sold['purchase_after_close_flag'].sum():,} rows")
print(f"   negative_timeline_flag (either): {sold['negative_timeline_flag'].sum():,} rows")


# ----------------------------
# Geographic data checks
# ---------------------------
def add_geo_flags(df):
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return df
    
    df['missing_coords_flag'] = df['Latitude'].isnull() | df['Longitude'].isnull()
    df['zero_coords_flag'] = (df['Latitude'] == 0) | (df['Longitude'] == 0)
    # California longitude should be negative, if positive, likely an error
    df['longitude_sign_flag'] = df['Longitude'] > 0

    # Rough CA bounding box as check against implausible coordinates
    ca_lat_range = (32.0, 42.5)
    ca_lon_range = (-125.0, -114.0)
    df['implausible_coords_flag'] = ~(
        df['Latitude'].between(*ca_lat_range) &
        df['Longitude'].between(*ca_lon_range)
    ) & ~df['missing_coords_flag']

    return df

sold = add_geo_flags(sold)
listings = add_geo_flags(listings)

geo_flag_cols = ['missing_coords_flag', 'zero_coords_flag', 'longitude_sign_flag', 'implausible_coords_flag']

print("\nGeographic data quality flags (sold dataset):")
for flag in geo_flag_cols:
    if flag in sold.columns:
        print(f"   {flag}: {sold[flag].sum():,} rows")
    else:
        print(f"   {flag}: skipped (Latitude/Longitude not found)")


# -----------------------
# Save cleaned datasets
# -----------------------
sold.to_csv('sold_cleaned.csv', index=False)
listings.to_csv('listings_cleaned.csv', index=False)

print("\n" + "=" * 70)
print("FINAL ROW COUNTS")
print("=" * 70)
print(f"  sold_cleaned.csv:           {len(sold):,} rows, {sold.shape[1]} columns")
print(f"  listings_cleaned.csv:       {len(listings):,} rows. {listings.shape[1]} columns")
print("\nNote: no rows were delted blased on flags in this script - all invalid/inconsistent records are flagged,")
print("not removed, per the handbook. Decide in Week 7 (IQR outlier pass) whether flagged records should be excluded.")
