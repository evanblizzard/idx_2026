import pandas as pd
import os

# --- Week 2-3: FRED ---

output_path = 'data/processed/'

# loading my combined datasets
sold = pd.read_csv(output_path + 'sold.csv', low_memory=False)
listings = pd.read_csv(output_path + 'listings.csv', low_memory=False)
# fetching mortgage rate data from FRED
mortgage_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US'
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
sold_nulls = sold_with_rates['rate_30yr_fixed'].isnull().sum()
listings_nulls = listings_with_rates['rate_30yr_fixed'].isnull().sum()

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
