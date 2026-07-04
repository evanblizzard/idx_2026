import pandas as pd
import glob
import os


# --- LISTINGS COMBINATION ---

data_path = 'data/raw/'
output_path = 'data/processed/'

# Find all listing files
listing_files = sorted(glob.glob(data_path + 'CRMLSListing*.csv'))
print(f"Found {len(listing_files)} listing files:")
for f in listing_files:
    print(f" {f}")


# Loading and Combining
dfs = []
for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    dfs.append(df)

listings = pd.concat(dfs, ignore_index=True)
print(f"\nAfter concat: {len(listings)} rows")

# Filter to just residential
listings = listings[listings['PropertyType'] == 'Residential']
print(f"After Residential filter: {len(listings)} rows")

# Saving
listings.to_csv(output_path + 'listings.csv', index=False)
print(f"\nSaved to {output_path}listings.csv")