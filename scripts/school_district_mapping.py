import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
sold = pd.read_csv(os.path.join(DATA_DIR, 'sold_cleaned.csv'))
listings = pd.read_csv(os.path.join(DATA_DIR, 'listings_cleaned.csv'))

# read california school district boundary geojson
school_districts = gpd.read_file("data/DistrictAreas.geojson")

print(f"Loaded {school_districts.shape[0]} district polygons, {school_districts.shape[1]} columns")
print(f"CRS: {school_districts.crs}")

unified_districts = school_districts[school_districts["DistrictType"] == "Unified"].copy()

print(f"Filtered to {len(unified_districts)} Unified Districts "
      f"(types available: {sorted(school_districts['DistrictType'].unique())})")

def add_school_district(df, unified_districts):
    df = df.copy()

    # ensuring a row has usable coordinates
    has_coords = df['Latitude'].notna() & df['Longitude'].notna()
    # checking for coords and rows WITHOUT the implausible coords flag
    if 'implausible_coords_flag' in df.columns:
        valid = has_coords & ~df['implausible_coords_flag']
    else:
        valid = has_coords

    # create new column filled with missing values
    df['DistrictName'] = pd.NA
    # if there are now valid rows, abort and return the df
    if valid.sum() == 0:
        return df

    # give me only rows where the valid is true
    sub = df.loc[valid]
    # turn longitude and latitude columns into point geometry objects
    geometry = gpd.points_from_xy(sub['Longitude'], sub['Latitude'])
    # make a geodataframe
    # sub[[]] means sub but with none of the original columns - just something to hold the geometry 
    gdf = gpd.GeoDataFrame(sub[[]], geometry=geometry, crs="EPSG:4326")
    # unified_districts.crs is EPSG:3857, so im converting
    gdf = gdf.to_crs(unified_districts.crs)

    # spatial join to match rows based on geometry
    # for every pont in gdf, it checks if it is within a polygon in "unified_districts"
    # left means keep every row from gdf even if no polygon contains it
    joined = gpd.sjoin(
        gdf,
        unified_districts[["DistrictName", "geometry"]],
        how="left",
        predicate="within"
    )

    # sjoin can produce dupe rows if a point sits on a shared boundary
    # between 2 district pollgyons; keeping first match per original row
    joined = joined[~joined.index.duplicated(keep="first")]


    # everything above (sub, gdf, joined) only ever contained the "valid rows"
    # this takes those results and puts them back in the right spots in the original full sized df
    df.loc[valid, "DistrictName"] = joined["DistrictName"].values
    return df

sold = add_school_district(sold, unified_districts)
listings = add_school_district(listings, unified_districts)

# Sanity Checks
print("\n=== Match rates ===")
print(f"sold:     {sold['DistrictName'].notna().sum():,} / {len(sold):,} matched")
print(f"listings: {listings['DistrictName'].notna().sum():,} / {len(listings):,} matched")
 
for label, df in [("sold", sold), ("listings", listings)]:
    valid_coords = df['Latitude'].notna() & df['Longitude'].notna()
    if 'implausible_coords_flag' in df.columns:
        valid_coords &= ~df['implausible_coords_flag']
    unmatched_valid = df[valid_coords & df['DistrictName'].isna()]
    print(f"\n[{label}] {len(unmatched_valid):,} rows had usable coordinates but no district match "
          f"(likely non-Unified areas, or points just outside CDE's mapped boundaries)")
    if len(unmatched_valid) > 0:
        print(unmatched_valid[['Latitude', 'Longitude']].describe())


# saving new datasets
sold.to_csv(os.path.join(DATA_DIR, 'sold_with_district.csv'), index=False)
listings.to_csv(os.path.join(DATA_DIR, 'listings_with_district.csv'), index=False)
print("\nSaved sold_with_district.csv and listings_with_district.csv")
