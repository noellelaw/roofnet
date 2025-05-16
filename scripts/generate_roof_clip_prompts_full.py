# GOAL: Generate a CSV file with image paths and prompts for roof classification dataset
# This script reads images from 'train' and 'val' subfolders in the provided dataset directory,
# extracts city names and material labels from filenames and folder names,
# and saves the result to a CSV file including a "split" column.
# Run: python generate_roof_clip_prompts.py \
# --dataset_dir "/path/to/RoofClassificationData_VLM" \
# --output_csv "/path/to/save/roof_dataset_clip_prompts_all.csv"
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

def generate_dataset_csv(dataset_root, output_csv_path):
    dataset_entries = []

    for split in ['train', 'val']:
        split_path = Path(dataset_root) / split
        if not split_path.exists():
            print(f"Split folder '{split}' not found in {dataset_root}")
            continue

        for material_folder in os.listdir(split_path):
            material_path = split_path / material_folder
            if not material_path.is_dir() or material_folder not in material_descriptions:
                continue

            description = material_descriptions[material_folder]

            for img_file in os.listdir(material_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    city = extract_city_from_filename(img_file)
                    if city:
                        prompt = f"{description} in {city}"
                        relative_path = os.path.join(split, material_folder, img_file)
                        dataset_entries.append({
                            'image': relative_path,
                            'prompt': prompt,
                            'split': split
                        })

    dataset_df = pd.DataFrame(dataset_entries)
    dataset_df.to_csv(output_csv_path, index=False)
    print(f"Dataset CSV saved to {output_csv_path} with {len(dataset_df)} entries")

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP-style prompt CSV for RoofNet dataset.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the RoofNet dataset root folder (containing 'train' and 'val')")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the generated CSV file")

    args = parser.parse_args()
    generate_dataset_csv(args.dataset_dir, args.output_csv)
