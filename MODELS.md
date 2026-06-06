# DepthForge - Downloading and Converting MiDaS and DPT Models

## Downloading MiDaS Models

### Method 1: Downloading from Hugging Face

```bash
# Download MiDaS model from Hugging Face
git lfs install
git clone https://huggingface.co/intel/MiDaS-small
git clone https://huggingface.co/intel/MiDaS-large

# Alternatively using huggingface_hub
pip install huggingface_hub
```

### Method 2: Downloading from the official website

```bash
# Download from the official MiDaS website
wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small.onnx
wget https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21.onnx
```

## Converting Models to OpenVINO Format

### Requirements:
- `openvino` (installed)
- `onnx` (installed)
- `openvino-dev` (optional)

### Conversion procedure:

```bash
# Convert MiDaS model to OpenVINO format
mo --input_model midas_v21.onnx \
   --output_dir ./models/midas \
   --model_name midas_depth \
   --compress_to_fp16

# Convert DPT model
mo --input_model dpt_large.onnx \
   --output_dir ./models/dpt \
   --model_name dpt_depth \
   --compress_to_fp16
```

## Python conversion example:

```python
import openvino as ov
import onnx
import numpy as np

def convert_midas_to_openvino(onnx_model_path, output_dir):
    """
    Convert MiDaS model to OpenVINO format
    """
    try:
        # Load ONNX model
        model = onnx.load(onnx_model_path)

        # Convert to OpenVINO
        core = ov.Core()
        ov_model = core.read_model(model=onnx_model_path)

        # Save OpenVINO model
        ov.serialize(ov_model, f"{output_dir}/midas_depth.xml")
        ov.serialize(ov_model, f"{output_dir}/midas_depth.bin")

        print(f"MiDaS model converted to OpenVINO in {output_dir}")
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

# Usage example:
# convert_midas_to_openvino("midas_v21.onnx", "./models/midas")
```

## Required models:

1. **MiDaS_small** - smaller model, faster
2. **MiDaS_large** - larger model, more accurate
3. **DPT_Large** - modern Dense Prediction Transformer model
4. **DPT_Hybrid** - compromise between quality and speed

## Directory structure after conversion:

```
models/
├── midas/
│   ├── midas_depth.xml
│   └── midas_depth.bin
├── dpt/
│   ├── dpt_depth.xml
│   └── dpt_depth.bin
└── config/
    └── model_config.json
```

## Usage in DepthForge:

After conversion, the models can be used in DepthForge:

```python
# In config.json set:
{
  "model": {
    "depth_estimation": {
      "method": "midas_like",
      "model_path": "./models/midas/midas_depth.xml",
      "device": "CPU",
      "resolution": "high",
      "use_openvino": true
    }
  }
}
```

## Alternative approach:

If models cannot be downloaded, DepthForge can work with synthetic algorithms (current implementation) and implement support for OpenVINO models in the future.
