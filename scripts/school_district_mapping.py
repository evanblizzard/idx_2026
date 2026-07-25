import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
sold = pd.read_csv(os.path.join(DATA_DIR, 'sold_with_rates.csv'))
listings = pd.read_csv(os.path.join(DATA_DIR, 'listings_with_rates.csv'))

# read california school district boundary geojson
school_districts = gpd.read_file("data/DistrictAreas.geojson")

print(school_districts.shape)
print(school_districts.columns)
print(school_districts.crs)
school_districts.head()

unified_districts = school_districts[school_districts["DistrictType"] == "Unified"]
print(unified_districts.shape)
unified_districts["DistrictType"].unique()


def add_school_district(df, unified_districts):
    df = df.copy()

    df["geometry"] = df.apply(
        lambda row: Point(row["Longitude"], row["Latitude"]), axis=1
    )
    # Correct: label as EPSG:4326 because that's what the raw values actually are
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # Now actually transform the coordinates to match unified_districts' CRS
    gdf = gdf.to_crs(unified_districts.crs)
    gdf.geometry.iloc[0]
    joined = gpd.sjoin(
        gdf,
        unified_districts[["DistrictName", "geometry"]],
        how="left",
        predicate="within"
    )

    df["DistrictName"] = joined["DistrictName"].values
    return df

sold = add_school_district(sold, unified_districts)
listings = add_school_district(listings, unified_districts)

# Sanity Check
print("Sold - missing district:", sold["DistrictName"].isna().sum())
print("Listing - missing district:", listings["DistrictName"].isna().sum())

print(len(sold), len(listings))
print(sold["DistrictName"].notna().sum(), "/", len(sold), "matched")
print(listings["DistrictName"].notna().sum(), "/", len(listings), "matched")

print(sold[["Latitude", "Longitude"]].isna().sum())
print(listings[["Latitude", "Longitude"]].isna().sum())

from shapely.geometry import Point
test_point = gpd.GeoDataFrame(
    {"geometry": [Point(-118.2437, 34.0522)]},
    crs="EPSG:4326"
).to_crs(unified_districts.crs)

match = gpd.sjoin(test_point, unified_districts[["DistrictName", "geometry"]], how="left", predicate="within")
print(match)

# rows with valid coordinates but no district match
unmatched_valid = sold[sold["DistrictName"].isna() & sold["Latitude"].notna()]
print(unmatched_valid[["Latitude", "Longitude"]].describe())
print(unmatched_valid[["Latitude", "Longitude"]].sample(10))

CA_LAT_MIN, CA_LAT_MAX = 32.0, 42.5
CA_LON_MIN, CA_LON_MAX = -125.0, -113.5

bad_coords = unmatched_valid[
    (unmatched_valid["Latitude"] < CA_LAT_MIN) | (unmatched_valid["Latitude"] > CA_LAT_MAX) |
    (unmatched_valid["Longitude"] < CA_LON_MIN) | (unmatched_valid["Longitude"] > CA_LON_MAX)
]
print(len(bad_coords), "rows have coordinates outside California's bounding box")
print(bad_coords[["Latitude", "Longitude"]].head(20))