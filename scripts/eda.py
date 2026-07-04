import pandas as pd
import os

# --- Week 2-3: EDA ---

output_path = 'data/processed/'

# loading my combined datasets
sold = pd.read_csv(output_path + 'sold.csv', low_memory=False)
listings = pd.read_csv(output_path + 'listings.csv', low_memory=False)

# --- Section 1: Basic Shape of Data ---
print("=== SOLD ===")
print(f"Rows: {sold.shape[0]}")
print(f"Columns: {sold.shape[1]}")


print("\n=== LISTINGS ===")
print(f"Rows: {listings.shape[0]}")
print(f"Columns: {listings.shape[1]}")

# --- Section 2: Data Types ---
print("\n=== SOLD DATA TYPES ===")
print(sold.dtypes)

print("\n=== LISTINGS DATA TYPES ===")
print(listings.dtypes)


# --- Section 3: Missing Value Analysis ---
def missing_report(df, name):
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({
        'missing_count': missing, 
        'missing_pct': pct
    }).sort_values('missing_pct', ascending=False)

    print(f"\n=== {name} MISSING VALUE REPORT ===")
    print(report[report['missing_count'] > 0].to_string())

    # flagging columns above 90% missing
    high_missing = report[report['missing_pct'] > 90]
    print(f"\nColumns above 90% missing in {name}: {len(high_missing)}")
    print(high_missing.to_string())

missing_report(sold, 'SOLD')
missing_report(listings, 'LISTINGS')

# TaxYear, TaxAnnualAmount, AboveGradeFinishedArea, FireplacesTotal, ElementarySchoolDistrict, CoveredSpaces, BusinessType,
# MiddleOrJuniorSchoolDistrict are completely empty everywhere

# noticing duplicate columns in listings: BUyerOfficeName and BuyerOfficeName.1, CloseData, and CloseDate.1, Latitutde and Latitude.1

# ClosePrice 75% missing in listings - makes sense

# Latitude/Longitude 13% missing in listings


# --- SECTION 4: Numeric Distribution Summary ---
key_fields = ['ClosePrice', 'LivingArea', 'DaysOnMarket']

for field in key_fields:
    print(f"\n===SOLD: {field} ===")
    print(sold[field].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# ---close price---
# Median $825k, makes sense, California
# Mean is 1.19M, being pulled up by expensive properties
# Min is $0, something wrong there
# Max is $989,500,000, huge outlier
# 99th percentile is $5.57M so ^ is crazy

# ---living area---
# Median 1644 sq ft
# min is 0, invalid
# max is 17,021,320 sq ft which is way too huge and must be an error
# 99th percentile is 5283 sq ft

# ---days on market---
# median 18 days 
# min -288, data entry error?
# max is 12,430 days, which would be 34 years on the market lol
# mean of 37 and median 18 shows right skew

# --- Section 3: Property Types ---
print(sold['PropertyType'].unique())
print(listings['PropertyType'].unique())

# deliverable for weeks 2-3 asks for filtering logic applied
# filtering logic for property types was applied in week 1 listings.py and sold.py

# --- Section 5: Mortgage Rate Enrichment (FRED) ---

# fetching mortgage rate data from FRED
mortgage_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTAGE30US'
mortgage = pd.read_csv(mortgage_url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

print(f"\nFetched {len(mortgage)} weekly mortgage rate records")
print(f"Date range: {mortgage['date'].min()} to {mortgage['date'].max()}")

# resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
print(f"Resampled to {len(mortgage_monthly)} monthly records")

# Creating matching year_month key on the MLS datasets
sold['CloseDate'] = pd.to_datetime(sold['CloseDate'])
sold['year_month'] = sold['CloseDate'].dt.to_period('M')

listings['ListingContractDate'] = pd.to_datetime(listings['ListingContractDate'])
listings['year_month'] = listings['ListingContractDate'].dt.to_period('M')

# Merging
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')

# Validating merge
sold_nulls = sold_with_rates['rate_30yr_fixed'].is_null().sum()
listings_nulls = listings_with_rates['rate_30yr_fixed'].is_null().sum()

print(f"\n=== MERGE VALIDATION ===")
print(f"Sold - rows with missing rate: {sold_nulls}")
print(f"Listings - rows with missing rate: {listings_nulls}")

if sold_nulls > 0 or listings_nulls > 0:
    print("Warning: Unmatched rows detected. Investigating date range mismatch...")
    print("Sold year_month range", sold['year_month'].min(), "-", sold['year_month'].max())
    print("Mortgage year_month range:", mortgage_monthly['year_month'].min(), "-", mortgage_monthly['year_month'].max())
else:
    print("Validation passed: no null rate values after merge")

# preview
print("\n=== Preview: Sold With Rates ===")
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']])

# Saving datasets
sold_with_rates.to_csv(output_path + 'sold_with_rates.csv', index=False)
listings_with_rates.to_csv(output_path + 'listings_with_rates.csv', index=False)
print("\nSaved sold_with_rates.csv and listings_with_rates.csv")



