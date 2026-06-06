# DepthForge - Model Integration Guide

## DPT Model Integration with OpenVINO

### Requirements:
- OpenVINO with conversion tools
- Python with libraries: torch, transformers, onnx, onnxscript

### Step by step:

1. **Prepare the environment:**
   ```bash
   pip install torch transformers onnx onnxscript
   ```

2. **Convert the DPT model to ONNX:**
   ```python
   import torch
   from transformers import DPTForDepthEstimation

   # Load the model
   model = DPTForDepthEstimation.from_pretrained('models/dpt')

   # Prepare dummy input
   dummy_input = torch.randn(1, 3, 384, 384)

   # Convert to ONNX
   torch.onnx.export(
       model,
       dummy_input,
       'models/dpt/dpt_large.onnx',
       export_params=True,
       opset_version=11,
       do_constant_folding=True,
       input_names=['input'],
       output_names=['output']
   )
   ```

3. **Convert ONNX to OpenVINO:**
   ```bash
   mo --input_model models/dpt/dpt_large.onnx \
      --output_dir models/dpt \
      --model_name dpt_depth \
      --compress_to_fp16
   ```

4. **Update configuration:**
   ```json
   {
     "model": {
       "depth_estimation": {
         "model_path": "./models/dpt/dpt_depth.xml",
         "use_openvino": true
       }
     }
   }
   ```

5. **Run the project:**
   ```bash
   python src/depth_forge.py --input test_image.jpg --output depth.png
   ```

### Output files after conversion:
- `models/dpt/dpt_depth.xml`
- `models/dpt/dpt_depth.bin`

### Testing:
```bash
python src/depth_forge.py --input data/test_image.jpg --output output/test_depth.png
```

### Alternative:
If you cannot convert the model, the project still works with the synthetic algorithm (default).
