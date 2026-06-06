#!/usr/bin/env python3
"""
Convert ONNX models to OpenVINO IR format (.xml + .bin)
Supported models:
  - models/dpt/dpt_large.onnx  -> models/dpt/openvino/
  - models/midas/midas_v21_small_256.onnx -> models/midas/openvino/
"""

import sys
from pathlib import Path

try:
    import openvino as ov
    from openvino import convert_model, save_model
    print(f"OpenVINO version: {ov.__version__}")
except ImportError:
    print("Error: OpenVINO is not installed.")
    print("Install with: pip install openvino")
    sys.exit(1)

# Project root directory (parent of this script's directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "DPT Large",
        "onnx_path": PROJECT_ROOT / "models" / "dpt" / "dpt_large.onnx",
        "output_dir": PROJECT_ROOT / "models" / "dpt" / "openvino",
        "model_name": "dpt_large",
        "compress_fp16": True,
    },
    {
        "name": "MiDaS v2.1 Small",
        "onnx_path": PROJECT_ROOT / "models" / "midas" / "midas_v21_small_256.onnx",
        "output_dir": PROJECT_ROOT / "models" / "midas" / "openvino",
        "model_name": "midas_v21_small_256",
        "compress_fp16": True,
    },
]


def convert_onnx_to_openvino(name: str, onnx_path: Path, output_dir: Path,
                              model_name: str, compress_fp16: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"  Source:  {onnx_path}")
    print(f"  Target:  {output_dir}/{model_name}.xml")
    print(f"  FP16:    {compress_fp16}")
    print(f"{'='*60}")

    if not onnx_path.exists():
        print(f"  [SKIPPED] ONNX file does not exist: {onnx_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"{model_name}.xml"

    try:
        print("  Loading ONNX model...")
        ov_model = convert_model(str(onnx_path))

        if compress_fp16:
            print("  Compressing to FP16 on save...")

        print(f"  Saving OpenVINO IR model -> {xml_path}")
        save_model(ov_model, str(xml_path), compress_to_fp16=compress_fp16)

        # Check output files
        bin_path = output_dir / f"{model_name}.bin"
        if xml_path.exists():
            xml_size = xml_path.stat().st_size / 1024
            print(f"  ✓ {xml_path.name}  ({xml_size:.1f} KB)")
        if bin_path.exists():
            bin_size = bin_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {bin_path.name}  ({bin_size:.1f} MB)")

        print(f"  [OK] Conversion of {name} completed successfully!")
        return True

    except Exception as e:
        print(f"  [ERROR] Conversion of {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Converting ONNX models -> OpenVINO IR")
    print("======================================\n")

    results = {}
    for model_cfg in MODELS:
        ok = convert_onnx_to_openvino(**model_cfg)
        results[model_cfg["name"]] = ok

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ ERROR / SKIPPED"
        print(f"  {status}  {name}")
    print(f"{'='*60}")

    if all(results.values()):
        print("\nAll models converted successfully!")
        print("You can now use the OpenVINO models from:")
        print("  models/dpt/openvino/dpt_large.xml")
        print("  models/midas/openvino/midas_v21_small_256.xml")
    else:
        print("\nSome conversions failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
