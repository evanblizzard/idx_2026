import pandas as pd
import os

# --- SOLD AGGREGATION ---

# My workflow:
# Loading all monthly sold CSVS and concatenating into one dataset
# For months with both regular and _filled versions, using the regular only
# For months with only _filled, using that and dropping the two extra columns at the end

data_path = 'data/raw/'
output_path = 'data/processed/'

# explicitly picking one file per month
sold_files = [
    'CRMLSSold202401.csv',
    'CRMLSSold202402.csv',
    'CRMLSSold202403.csv',
    'CRMLSSold202404.csv',
    'CRMLSSold202405_filled.csv',  # only filled available
    'CRMLSSold202406_filled.csv',  # only filled available
    'CRMLSSold202407_filled.csv',  # only filled available
    'CRMLSSold202408.csv',
    'CRMLSSold202409.csv',
    'CRMLSSold202410.csv',
    'CRMLSSold202411.csv',
    'CRMLSSold202412.csv',
    'CRMLSSold202501_filled.csv',  # only filled available
    'CRMLSSold202502.csv',
    'CRMLSSold202503.csv',
    'CRMLSSold202504.csv',
    'CRMLSSold202505.csv',
    'CRMLSSold202506.csv',
    'CRMLSSold202507.csv',
    'CRMLSSold202508.csv',
    'CRMLSSold202509.csv',
    'CRMLSSold202510.csv',
    'CRMLSSold202511.csv',
    'CRMLSSold202512.csv',
    'CRMLSSold202601.csv',
    'CRMLSSold202602.csv',
    'CRMLSSold202603.csv',
    'CRMLSSold202604.csv',
]

print(f"Loading {len(sold_files)} sold files:")

dfs = []
for fname in sold_files:
    fpath = data_path + fname
    df = pd.read_csv(fpath, low_memory=False)

    # Dropping extra columns from _filled
    if '_filled' in fname:
        df = df.drop(columns=['latfilled', 'lonfilled'], errors='ignore')
        print(f" {fname}: {len(df)} rows (dropped latfilled, lonfilled)")
    else:
        print(f" {fname}: {len(df)} rows")

    dfs.append(df)

sold = pd.concat(dfs, ignore_index=True)
print(f"\nAfter concat: {len(sold)} rows")

# Filter to residential only
sold = sold[sold['PropertyType'] == 'Residential']
print(f"After Residential filter: {len(sold)} rows")

# saving
sold.to_csv(output_path + 'sold.csv', index=False)
print(f"\nSaved to {output_path}sold.csv")

