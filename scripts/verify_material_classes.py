import os
import pandas as pd
from pathlib import Path
import shutil
import ast

# === CONFIG ===
dataset_dir = "PATH/TO/ROOFNET"  # Replace with your full path
city_materials_csv = "resources/City_Roof_Materials_with_Continent_and_Country_Centroids.csv"

# === LOAD CSV AND BUILD LOOKUP ===
df = pd.read_csv(city_materials_csv)
df['city_key'] = df['City'].str.lower().str.replace(" ", "_").str.replace(",", "")
city_to_materials = {}

# Parse the 'Roof Materials' JSON-like strings into Python lists
for _, row in df.iterrows():
    try:
        city = row['city_key']
        material_list = [m['class'] for m in ast.literal_eval(row['Roof Materials'])]
        city_to_materials[city] = material_list
    except Exception as e:
        print(f"Error parsing materials for city: {row['City']}: {e}")

# === CITY PARSER FUNCTION ===
def extract_city_from_filename(filename):
    base = Path(filename).stem
    if '-' in base:
        city_part = base.split('-')[0]
    elif 'height' in base:
        city_part = base.split('_height')[0]
    elif 'imsat' in base:
        city_part = base.split('_imsat')[0]
    else:
        return None
    return city_part.replace('_', ' ').lower().replace(",", "").strip()

# === MAIN VALIDATION SCRIPT ===
for material_folder in os.listdir(dataset_dir):
    material_path = os.path.join(dataset_dir, material_folder)
    if not os.path.isdir(material_path):
        continue

    reassess_dir = os.path.join(material_path, "reassess")
    os.makedirs(reassess_dir, exist_ok=True)

    for img_file in os.listdir(material_path):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            city = extract_city_from_filename(img_file)
            if not city:
                continue

            allowed_materials = city_to_materials.get(city)
            if allowed_materials is None:
                print(f"No material list found for city: {city}")
                continue

            if material_folder not in allowed_materials:
                src_path = os.path.join(material_path, img_file)
                dst_path = os.path.join(reassess_dir, img_file)
                shutil.move(src_path, dst_path)
                print(f"Moved {img_file} from {material_folder} to reassess (city: {city})")
