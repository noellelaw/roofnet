# RoofNet
Version: 1.0 License: CC BY-NC 4.0 (with ODbL terms for derivative geospatial data) 
Citation: RoofNet: A Global Multimodal Dataset for Roof Material Classification, (under review)
Dataset Download: https://www.kaggle.com/datasets/noellelaw/roofnet

## Overview 
RoofNet is the largest and most geographically diverse open-access dataset for global roof material classification. It consists of high-resolution Earth Observation (EO) image tiles paired with structured metadata and curated textual prompts describing roofing characteristics across 14 material classes and an "Unknown" class. The dataset is designed to support hazard preparedness, resilience planning, and post-disaster supply-chain analysis.

RoofNet includes:
-51,503 EO image tiles spanning 184 urban regions across 112 countries
-14 roof material classes (e.g., Asphalt Tiles, Clay Tiles, Metal Sheets, Thatch, etc.)
-A multimodal CSV file with rich per-sample metadata
-RemoteCLIP ViT-L/14 models fine-tuned for roof classification
-Training and evaluation code for reproducible fine-tuning and VLM experimentation. Code is available here: 


RoofNet includes 14 roofing material classes grouped into 5 categories:
Natural/Traditional: Thatch, Green Vegetative
Stone/Ceramic Tiles: Stone Slates, Clay Tiles
Asphalt/Concrete/Wood Tiles: Asphalt Tiles, Concrete Tiles, Wood Tiles
Sheet-Based: Metal Sheet Materials, Polycarbonate Sheet Materials, Glass Sheet Materials
Synthetic/Amorphous: Amorphous Asphalt, Amorphous Concrete, Amorphous Membrane, Amorphous Fabric

Fine-Tuned Models The models/ folder includes a fine-tuned version of RemoteCLIP ViT-L/14, adapted using 6,000 manually annotated samples, class re-balancing, and with prompts incorporating geographic and material cues. See the notebooks folder for reproducible training and evaluation pipelines.
