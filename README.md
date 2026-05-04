# AI-Based Automatic Road Network Extraction from Satellite Imagery

Lightweight local project for extracting roads from satellite images and converting them into a graph/network representation.

The main pipeline is:

1. Load satellite image
2. Predict road mask with a lightweight Keras U-Net, or use a classical fallback
3. Improve broken road continuity with image processing
4. Skeletonize roads
5. Convert the skeleton to a graph
6. Export masks, graph files, and GIS-friendly GeoJSON

This is designed for VS Code, Python virtual environments, and CPU-friendly execution on moderate laptops.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the smoothest TensorFlow setup, use Python 3.10, 3.11, or 3.12. If TensorFlow installation is heavy on your laptop, you can still run the graph-generation pipeline with the classical fallback:

```powershell
pip install -r requirements-light.txt
```

## Quick Run

```powershell
python scripts/create_synthetic_sample.py
python -m road_extractor.cli extract --image data/sample/input.png --output outputs/demo --no-model
```

With a trained model:

```powershell
python -m road_extractor.cli extract --image data/sample/input.png --output outputs/demo --weights models/road_unet.keras
```

## Train A Small Model

Prepare folders:

```text
data/train/images
data/train/masks
data/val/images
data/val/masks
```

Images and masks should have matching filenames. Masks should be binary road masks where roads are white and background is black.

For a quick TensorFlow test before using real satellite data, create a tiny synthetic dataset:

```powershell
python scripts/create_synthetic_dataset.py
```

Run:

```powershell
python -m road_extractor.cli train --train-images data/train/images --train-masks data/train/masks --val-images data/val/images --val-masks data/val/masks --output models/road_unet.keras --epochs 20
```

## Outputs

For an input image, the extractor writes:

- `raw_mask.png`: initial predicted/fallback road mask
- `repaired_mask.png`: continuity-improved mask
- `skeleton.png`: thin road centerline
- `road_graph.graphml`: graph/network representation
- `road_graph.geojson`: GIS-friendly vector output in image pixel coordinates
- `overlay.png`: visual road overlay

## Why This Is More Than Segmentation

Most basic road extraction projects stop at the mask. This project adds:

- morphological closing and bridge repair for broken roads
- skeletonization to produce road centerlines
- graph construction with junctions and edges
- GraphML and GeoJSON export for mapping, navigation, and smart city analysis

## Suggested Project Structure

```text
road_extractor/
  cli.py
  config.py
  data.py
  graph.py
  model.py
  pipeline.py
  postprocess.py
  visualize.py
```

## Notes

- Coordinates in GeoJSON are pixel coordinates by default.
- If you know the image georeferencing transform, you can later extend `graph.py` to convert pixel coordinates into real-world map coordinates.
- Keep images small during development, for example 256x256 or 512x512 tiles.
