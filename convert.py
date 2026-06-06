#!/usr/bin/env python3
"""
Convert DPT model to ONNX format
"""

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