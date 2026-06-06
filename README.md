# DepthForge

> **Depth perception from 2D images — tactile 3D reproductions for the visually impaired**

![Depth map comparison – Józef Chełmoński "Indian Summer"](assets/depth_comparison_preview.jpg)
*Depth map comparison for Józef Chełmoński's "Indian Summer" (Google Art Project).  
Left to right: original · Standard synthetic · OpenVINO MiDaS v2.1 Small · OpenVINO DPT Large · Ensemble (fusion of all methods)*

---

## Purpose

DepthForge analyses a flat 2D image and reconstructs the **depth (spatial) information** hidden within it — estimating which parts of the scene are close and which are distant.  
The resulting depth maps serve as the foundation for creating **3D tactile reproductions** that allow visually impaired people to physically explore artworks and museum exhibits through touch.

The workflow is two-stage:

1. **DepthForge** — automated depth extraction from the image using AI models (MiDaS, DPT) accelerated by Intel OpenVINO.  
2. **3D print preparation** — the depth data is handed to a 3D printing specialist who prepares it in the correct physical form for tactile exploration.  
   This stage is led by **Jakub Oleksy**, specialist in 3D print analysis:  
   [linkedin.com/in/jakub-oleksy-672668333/](https://www.linkedin.com/in/jakub-oleksy-672668333/)

> **GIMP Plugin** — the repository also contains a very early, preliminary integration layer for the GIMP image editor (`src/gimp_plugin.py`). The plugin is **not yet functional**; integration work is currently in progress.

---

## Description

DepthForge is a tool for generating depth maps from museum images, designed for creating 3D tactile visualizations for people with visual impairments.

This project enables processing of museum images to generate depth maps that can be used to create 3D tactile maps for people with visual impairments. It uses:
- OpenCV for image operations
- OpenVINO for efficient ML model processing
- PyTorch for image analysis
- Specialized algorithms for better depth visualization

## Requirements

- Python 3.8+
- OpenCV
- OpenVINO
- PyTorch
- NumPy
- SciPy
- Scikit-image
- numpy-stl
- transformers

## Installation

```bash
# Clone the repository
git clone https://github.com/GrzegorzOle/DepthForge.git
cd DepthForge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt

# Download OpenVINO models (DPT Large + MiDaS v2.1 Small)
python download_models.py
```

> **Note:** Model weights are not stored in the repository (they exceed GitHub's 100 MB file limit).  
> They are distributed as assets attached to the [GitHub Release](https://github.com/GrzegorzOle/DepthForge/releases/latest).  
> `download_models.py` fetches them automatically.

### Manual model download (optional)

```bash
# Download only DPT Large
python download_models.py --model dpt

# Download only MiDaS
python download_models.py --model midas

# Specify a different release
python download_models.py --release v0.1.0
```

## Usage

### Single image

```bash
python src/depth_forge.py --input input_image.jpg --output output_depth.png --enhanced-output enhanced_depth.png --tactile-output tactile_map.png
```

### Full pipeline depth maps + STL (demo - waiting for correct project off Kuba for 3D printing)

```bash
python src/depth_pipeline.py --input data/Stańczyk.jpg --output-dir output/stanczyk --width-mm 200 --relief-mm 12
```

### Batch processing

```bash
python src/depth_forge.py --batch --input-dir data/ --output-dir output/
```

### Benchmark (all methods + ensemble)

```bash
python benchmark.py
```

## Features specific to museums

- Depth map generation for museum images
- Optimization for 3D visualization
- Support for 3D printing (tactile maps) — binary STL export
- Integration with Braille systems and 3D visualization

## Depth map versions

The project generates three different versions of the depth map:
1. **Basic depth map** - generated based on image intensity
2. **Enhanced depth map** - with applied contrast techniques (CLAHE)
3. **Tactile map** - optimized for 3D printing for visually impaired people

## OpenVINO Integration

This project is designed to integrate with OpenVINO for improved depth estimation:
- Support for MiDaS and DPT models
- Efficient inference on CPU/GPU
- Integration with 3D printing workflows

## Project Structure

```
DepthForge/
├── assets/              # Static assets (preview images etc.)
├── config.json          # Project configuration
├── requirements.txt     # Required libraries
├── benchmark.py         # Benchmark – all methods + ensemble
├── src/                 # Source code
│   ├── depth_forge.py   # Core depth map generation
│   ├── depth_pipeline.py# Full pipeline: depth → STL
│   ├── advanced_3d_generator.py
│   └── gimp_plugin.py   # GIMP integration (work in progress)
├── data/                # Input data directory
├── models/              # ML models (MiDaS and DPT in OpenVINO format)
│   ├── midas/openvino/
│   └── dpt/openvino/
└── output/              # Output directory
```

## Configuration

Configuration is located in `config.json`:
- `model.depth_estimation`: Settings for depth estimation model
- `processing`: Image processing settings
- `tactile`: Tactile visualization settings

## For users interested in 3D visualization

This project is designed to create depth maps that can be used to:
1. Create 3D tactile maps for people with visual impairments
2. Visualize museum images in tactile form
3. Integrate with Braille systems and 3D visualization

## Development

The project can be extended with:
- Integration with specific depth estimation models (MiDaS, DPT)
- Support for different museum image formats
- Graphical interface
- Webcam support
- 3D printing system integration