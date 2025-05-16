
# GOAL: Generate metadata CSV for RoofNet images with additional information from a CSV file containing city, continent, and coordinates (look in resources).
# This script will read the images from the specified dataset directory, extract metadata from filenames, and save it to a CSV file.
# Run example: python generate_roof_material_metadata.py --dataset_dir <RoofNet_dataset_dir> --city_csv <city_coordinates_csv>
import os
import argparse
import shutil
from pathlib import Path
import pandas as pd
import re
from math import radians, cos, sin, asin, sqrt
import numpy as np

# === Haversine distance calculator ===
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def normalize_city(city):
    if pd.isna(city):
        return "unknown"
    return city.lower().replace(",", "").replace(" ", "_")


def main(base_dir, csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')
    df['city_key'] = df['City'].apply(normalize_city)
    city_to_continent = dict(zip(df['city_key'], df['Continent']))
    city_to_coords = dict(zip(df['city_key'], zip(df['Latitude'], df['Longitude'])))

    splits = ["train", "val"]
    material_folders = [
        'PolycarbonateSheetMaterials', 'AmorphousAsphalt', 'AmorphousConcrete', 'AmorphousFabric',
        'AmorphousMembrane', 'AsphaltTiles', 'ClayTiles', 'ConcreteTiles', 'GlassSheetMaterials',
        'GreenVegetative', 'MetalSheetMaterials', 'StoneSlates', 'Thatch', 'WoodTiles', 'Unknown'
    ]

    records = []
    count = 0

    for split in splits:
        parent_dir = os.path.join(base_dir, split)
        for folder in material_folders:
            folder_path = Path(parent_dir) / folder
            if not folder_path.exists():
                print(f"⚠️ Folder not found: {folder_path}")
                continue

            image_paths = [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in folder_path.glob(ext)]
            for image_path in image_paths:
                filename = image_path.name
                if len(image_path.stem.split("_height")) > 1:
                    city_key_raw = image_path.stem.split("_height")[0]
                elif len(image_path.stem.split("_imsat")) > 1:
                    city_key_raw = image_path.stem.split("_imsat")[0]
                else:
                    city_key_raw = image_path.stem.split("-")[0]
                city_key = city_key_raw.lower()

                # Extract lat/lon
                if 'imsat_' in image_path.stem:
                    latlong_str = image_path.stem.split('imsat_')[-1].split('_')[0]
                    match = re.match(r"(-?\d+\.\d{1,7})(-?\d+\.\d+)", latlong_str)
                    if not match or '-' in match.group(2):
                        match = re.match(r"(-?\d+\.\d{1,8})(-?\d+\.\d+)", latlong_str)
                    elif float(match.group(2)) > 180:
                        match = re.match(r"(-?\d+\.\d{1,8})(-?\d+\.\d+)", latlong_str)

                    if match:
                        lat, lon = float(match.group(1)), float(match.group(2))
                        # Haversine matching
                        closest_city, closest_dist = None, float("inf")
                        for row in df.itertuples():
                            city_lat, city_lon = row.Latitude, row.Longitude
                            dist = haversine(lat, lon, city_lat, city_lon)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_city = row.city_key
                                closest_lat, closest_lon = city_lat, city_lon
                        if closest_dist > 150:
                            if city_key in city_to_coords:
                                lat, lon = city_to_coords[city_key]
                            else:
                                lat, lon = closest_lat, closest_lon
                                city_key = closest_city
                                print(f"No city match in filename. Using closest city: {closest_city} ({closest_dist:.1f} mi)")
                    else:
                        print(f"Failed to parse lat/lon from filename: {image_path}")
                        continue
                else:
                    if city_key in city_to_coords:
                        lat, lon = city_to_coords[city_key]
                    else:
                        print(f"No lat/lon found for city '{city_key}' in CSV for file: {image_path}")
                        continue

                material_label = folder
                continent = city_to_continent.get(city_key, "Unknown")
                if continent == "Unknown":
                    print(f"No continent found for city_key: '{city_key}' (filename: {filename})")

                # Additional metadata
                height = numstories = roofshape = fpArea = np.nan
                parts = filename.replace(".jpg", "").split("_")
                for part in parts:
                    if part.startswith("height"):
                        val = part.replace("height", "")
                        height = float(val) if val != "NA" else np.nan
                    elif part.startswith("numstories"):
                        val = part.replace("numstories", "")
                        numstories = float(val) if val != "NA" else np.nan
                    elif part.startswith("roofshape"):
                        val = part.replace("roofshape", "")
                        roofshape = val if val != "NA" else np.nan
                    elif part.startswith("fpArea"):
                        val = part.replace("fpArea", "")
                        fpArea = float(val) if val != "NA" else np.nan
                if material_label in ["AmorphousMembrane", "AmorphousFabric", "AmorphousConcrete", "AmorphousAsphalt"]:
                    roofshape = "Flat"

                records.append({
                    "split": split,
                    "city": city_key,
                    "latitude": lat,
                    "longitude": lon,
                    "material_class": material_label,
                    "country": df.loc[df['city_key'] == city_key, 'Country'].values[0] if city_key in df['city_key'].values else "Unknown",
                    "continent": continent,
                    "filename": filename,
                    "height": height,
                    "numstories": numstories,
                    "roofshape": roofshape,
                    "fpArea": fpArea
                })
                count += 1

    output_df = pd.DataFrame(records)
    output_csv_path = os.path.join(base_dir, "roof_materials_augmented_all.csv")
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Combined metadata CSV saved to: {output_csv_path}")
    print(f"\nTotal records processed: {count:,}")

    # === Statistics
    print(f"\n📊 Dataset contains {len(output_df):,} total records\n")
    for field in ['fpArea', 'height', 'numstories', 'roofshape']:
        available = output_df[field].notna().sum()
        percent = (available / len(output_df)) * 100
        print(f"{field:<12}: {available:,} entries ({percent:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata CSV for RoofNet images.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the dataset base directory (with train/val subfolders)")
    parser.add_argument('--city_csv', type=str, required=True, help="Path to CSV with city, continent, and coordinates")
    args = parser.parse_args()

    main(args.dataset_dir, args.city_csv)