# GOAL: Generate a CSV file with image paths and prompts for roof classification dataset
# This script will read the images from the specified dataset directory, extract metadata from filenames,
# and save it to a CSV file. This can be used for training a RemoteCLIP model.
# How to run: python generate_roof_clip_prompts.py \
# --dataset_dir "/path/to/RoofClassificationData_VLM/train" \
# --output_csv "/path/to/save/roof_dataset_clip_prompts_subset.csv"
import os
import argparse
import pandas as pd
from pathlib import Path

# === MATERIAL DESCRIPTIONS ===
material_descriptions = {
    "Thatch": "thatched roof (dried grasses / straw or palm)",
    "GreenVegetative": "roof with vegetation on it",
    "StoneSlates": "dark stone slate roof",
    "ClayTiles": "tiled clay / tiled ceramic roof",
    "AsphaltTiles": "angled asphalt shingle roof",
    "ConcreteTiles": "tiled concrete / tiled cement roof",
    "WoodTiles": "wood shingle roof",
    "MetalSheetMaterials": "corrugated or tiled metal roof (silver / dark / painted)",
    "PolycarbonateSheetMaterials": "polycarbonate roof",
    "GlassSheetMaterials": "glass roof (clear or mirrored)",
    "AmorphousConcrete": "flat concrete roof",
    "AmorphousAsphalt": "asphalt-coated roof (bitumen layer or rolled roofing)",
    "AmorphousMembrane": "membrane roof (bright EPDM/TPO)",
    "AmorphousFabric": "tensile fabric roof (PVC / PTFE / canvas)",
    "Unknown": "unknown material, image may be too low resolution or obstructed"
}

# === EXTRACT CITY from filename ===
def extract_city_from_filename(filename):
    base = Path(filename).stem
    if '-' in base:
        city_part = base.split('-')[0]
    elif 'height' in base:
        city_part = base.split('_height')[0]
    elif 'imsat' in base:
        city_part = base.split('_imsat')[0]
    else:
        print(f"Could not parse city name from filename: {filename}")
        return None
    return city_part.replace('_', ' ').title()

# === MAIN FUNCTION ===
def generate_dataset_csv(parent_image_dir, output_csv_path):
    dataset_entries = []

    for material_folder in os.listdir(parent_image_dir):
        material_path = os.path.join(parent_image_dir, material_folder)
        if os.path.isdir(material_path) and material_folder in material_descriptions:
            description = material_descriptions[material_folder]

            for img_file in os.listdir(material_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    city = extract_city_from_filename(img_file)
                    if city:
                        prompt = f"{description} in {city}"
                        relative_path = os.path.join(material_folder, img_file)
                        dataset_entries.append({
                            'image': relative_path,
                            'prompt': prompt
                        })
        else:
            print(f"Skipping folder '{material_folder}' (not in material_descriptions)")

    dataset_df = pd.DataFrame(dataset_entries)
    dataset_df.to_csv(output_csv_path, index=False)
    print(f"Dataset CSV saved to {output_csv_path}")

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP-style prompt CSV for roof classification dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the root folder containing roof material subfolders")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the generated CSV file")

    args = parser.parse_args()
    generate_dataset_csv(args.dataset_dir, args.output_csv)