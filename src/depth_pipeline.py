#!/usr/bin/env python3
"""
DepthForge – Complete depth pipeline + STL for 3D printing (Prusa)

Steps:
  1. Load the input image
  2. Compute depth maps with three methods simultaneously:
       • Standard (synthetic)
       • OpenVINO MiDaS v2.1 Small
       • OpenVINO DPT Large
  3. Fuse (ensemble) depth maps with weights:
       DPT × 0.50 + MiDaS × 0.35 + Standard × 0.15
  4. Post-processing: CLAHE + Gaussian smoothing
  5. Export:
       • PNG depth maps (each method + ensemble)
       • INFERNO colour visualisations
       • Binary STL (watertight) ready for Prusa printing

Usage:
  python src/depth_pipeline.py --input data/Stańczyk.jpg
  python src/depth_pipeline.py --input data/Stańczyk.jpg --output-dir output/stanczyk_stl \\
      --width-mm 200 --relief-mm 12 --base-mm 3 --mesh-px 512
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from stl import mesh as stl_mesh
from stl.stl import Mode as StlMode

# ── Path to the DepthForge module ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from depth_forge import DepthForge

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH MAP FUSION
# ─────────────────────────────────────────────────────────────────────────────

# Method weights: DPT (highest quality) > MiDaS > Standard (synthetic)
ENSEMBLE_WEIGHTS = {
    "dpt":      0.50,
    "midas":    0.35,
    "standard": 0.15,
}


def normalize_f32(arr: np.ndarray) -> np.ndarray:
    """Normalizes a 2D array to the range [0, 1] float32."""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr.astype(np.float32) - a_min) / (a_max - a_min))


def fuse_depth_maps(depth_maps: dict) -> np.ndarray:
    """
    Weighted fusion of depth maps.

    Args:
        depth_maps: {'standard': ndarray, 'midas': ndarray, 'dpt': ndarray}
                    Each map can be uint8 or float – any range.

    Returns:
        Fused depth map float32 [0, 1]
    """
    acc        = None
    weight_sum = 0.0

    for key, depth in depth_maps.items():
        weight = ENSEMBLE_WEIGHTS.get(key, 0.0)
        if weight == 0.0 or depth is None:
            continue
        norm = normalize_f32(depth)
        if acc is None:
            acc = norm * weight
        else:
            acc += norm * weight
        weight_sum += weight

    if acc is None or weight_sum < 1e-9:
        raise ValueError("No depth maps available for fusion.")

    return (acc / weight_sum).astype(np.float32)


def postprocess_depth(depth_f32: np.ndarray,
                      clahe_clip: float = 2.5,
                      sigma: float = 1.5) -> np.ndarray:
    """
    Enhance the fused depth map:
      1. Gaussian smoothing – noise reduction
      2. CLAHE – local contrast enhancement

    Args:
        depth_f32: map [0, 1] float32
        clahe_clip: CLAHE contrast limit
        sigma: Gaussian standard deviation (pixels)

    Returns:
        Processed map uint8 [0, 255]
    """
    # Gaussian smoothing
    smoothed = gaussian_filter(depth_f32, sigma=sigma)

    # → uint8
    img_u8 = (normalize_f32(smoothed) * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    return clahe.apply(img_u8)


# ─────────────────────────────────────────────────────────────────────────────
#  STL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_stl(depth_u8: np.ndarray,
                 output_path: str,
                 width_mm:     float = 200.0,
                 relief_mm:    float = 10.0,
                 base_mm:      float = 3.0,
                 max_px:       int   = 512,
                 invert_depth: bool  = False,
                 flip_x:       bool  = False,
                 flip_y:       bool  = True) -> None:
    """
    Converts a depth map to a binary STL file (watertight) for 3D printing.

    The model consists of:
      • Top surface (relief) – triangle mesh with height from the depth map
      • Flat base plate
      • Four side walls closing the solid

    Physical dimensions:
      • Width:  width_mm  [mm]
      • Depth:  auto (preserves image aspect ratio)
      • Height: base_mm + relief_mm [mm]

    Args:
        depth_u8:     Depth map uint8 [0-255]
        output_path:  Output path (.stl)
        width_mm:     Model width [mm]
        relief_mm:    Maximum relief height above base [mm]
        base_mm:      Base plate thickness [mm]
        max_px:       Maximum mesh resolution (longer side) [px]
        invert_depth: Invert depth values (255-depth)
        flip_x:       Horizontal mirror
        flip_y:       Vertical mirror (default True –
                      corrects image→Prusa Slicer Y orientation)
    """
    # ── Orientation transforms ────────────────────────────────────────────────
    d = depth_u8.copy()
    if invert_depth:
        d = 255 - d
    if flip_x:
        d = np.fliplr(d)
    if flip_y:
        d = np.flipud(d)

    # ── Mesh scaling ──────────────────────────────────────────────────────────
    orig_h, orig_w = d.shape
    if max(orig_h, orig_w) > max_px:
        scale = max_px / max(orig_h, orig_w)
        new_w = max(2, int(orig_w * scale))
        new_h = max(2, int(orig_h * scale))
        depth_scaled = cv2.resize(d, (new_w, new_h),
                                  interpolation=cv2.INTER_LANCZOS4)
    else:
        new_w, new_h  = orig_w, orig_h
        depth_scaled  = d

    # ── Convert px → mm ───────────────────────────────────────────────────────
    height_mm = width_mm * new_h / new_w          # depth [mm]
    px_to_mm_x = width_mm  / (new_w - 1)
    px_to_mm_y = height_mm / (new_h - 1)

    depth_norm = depth_scaled.astype(np.float32) / 255.0  # [0, 1]

    logger.info(f"STL mesh: {new_w}×{new_h} px  →  "
                f"{width_mm:.1f}×{height_mm:.1f}×{base_mm + relief_mm:.1f} mm  "
                f"({2*(new_w-1)*(new_h-1)} top triangles + 12 side faces)")

    # ── Top surface ───────────────────────────────────────────────────────────
    rows, cols = new_h, new_w

    def vtx(c, r):
        """Mesh vertex coordinates [mm]."""
        return (
            c * px_to_mm_x,
            r * px_to_mm_y,
            base_mm + depth_norm[r, c] * relief_mm,
        )

    n_top = 2 * (rows - 1) * (cols - 1)

    # ── Side and bottom triangles (watertight) ────────────────────────────────
    # Each side wall: left, right, front, back (strip of triangles)
    # Bottom: 2 triangles

    n_side_left  = 2 * (rows - 1)
    n_side_right = 2 * (rows - 1)
    n_side_front = 2 * (cols - 1)
    n_side_back  = 2 * (cols - 1)
    n_bottom     = 2

    n_total = n_top + n_side_left + n_side_right + n_side_front + n_side_back + n_bottom

    all_triangles = np.zeros(n_total, dtype=stl_mesh.Mesh.dtype)
    idx = 0

    # -- Top surface ----------------------------------------------------------
    for r in range(rows - 1):
        for c in range(cols - 1):
            v00 = vtx(c,     r    )
            v10 = vtx(c + 1, r    )
            v01 = vtx(c,     r + 1)
            v11 = vtx(c + 1, r + 1)

            # Triangle A
            all_triangles['vectors'][idx] = [v00, v10, v01]
            idx += 1
            # Triangle B
            all_triangles['vectors'][idx] = [v10, v11, v01]
            idx += 1

    # Helper: bottom (z = 0) at corners
    def bot(c, r):
        return (c * px_to_mm_x, r * px_to_mm_y, 0.0)

    # -- Left wall (c = 0) ----------------------------------------------------
    for r in range(rows - 1):
        t = vtx(0, r);    t1 = vtx(0, r + 1)
        b = bot(0, r);    b1 = bot(0, r + 1)
        all_triangles['vectors'][idx] = [t, b, t1];   idx += 1
        all_triangles['vectors'][idx] = [b, b1, t1];  idx += 1

    # -- Right wall (c = cols-1) ----------------------------------------------
    for r in range(rows - 1):
        t = vtx(cols - 1, r);    t1 = vtx(cols - 1, r + 1)
        b = bot(cols - 1, r);    b1 = bot(cols - 1, r + 1)
        all_triangles['vectors'][idx] = [t, t1, b];   idx += 1
        all_triangles['vectors'][idx] = [b, t1, b1];  idx += 1

    # -- Front wall (r = 0) ---------------------------------------------------
    for c in range(cols - 1):
        t = vtx(c, 0);    t1 = vtx(c + 1, 0)
        b = bot(c, 0);    b1 = bot(c + 1, 0)
        all_triangles['vectors'][idx] = [t, t1, b];   idx += 1
        all_triangles['vectors'][idx] = [b, t1, b1];  idx += 1

    # -- Back wall (r = rows-1) -----------------------------------------------
    for c in range(cols - 1):
        t = vtx(c, rows - 1);    t1 = vtx(c + 1, rows - 1)
        b = bot(c, rows - 1);    b1 = bot(c + 1, rows - 1)
        all_triangles['vectors'][idx] = [t, b, t1];   idx += 1
        all_triangles['vectors'][idx] = [b, b1, t1];  idx += 1

    # -- Bottom ---------------------------------------------------------------
    bl = bot(0,        0       )
    br = bot(cols - 1, 0       )
    fl = bot(0,        rows - 1)
    fr = bot(cols - 1, rows - 1)
    all_triangles['vectors'][idx] = [bl, br, fl];  idx += 1
    all_triangles['vectors'][idx] = [br, fr, fl];  idx += 1

    # ── Compute normals and save ──────────────────────────────────────────────
    m = stl_mesh.Mesh(all_triangles)
    m.update_normals()
    m.save(output_path, mode=StlMode.BINARY)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"STL saved: {output_path}  ({size_mb:.1f} MB, {n_total} triangles)")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str,
                 output_dir: str     = "output",
                 width_mm:   float   = 200.0,
                 relief_mm:  float   = 10.0,
                 base_mm:    float   = 3.0,
                 mesh_px:    int     = 512,
                 clahe_clip: float   = 2.5,
                 smooth_sigma: float = 1.5,
                 invert_depth: bool  = False,
                 flip_x:       bool  = False,
                 flip_y:       bool  = True) -> None:
    """
    Runs the full pipeline:
      image → 3× depth map → fusion → STL

    Args:
        input_path:    Path to the input image
        output_dir:    Output directory
        width_mm:      STL model width [mm]
        relief_mm:     Maximum relief height [mm]
        base_mm:       Base plate thickness [mm]
        mesh_px:       Maximum STL mesh resolution [px]
        clahe_clip:    CLAHE clip limit
        smooth_sigma:  Gaussian smoothing sigma [px]
        invert_depth:  Invert depth values
        flip_x:        Horizontal mirror
        flip_y:        Vertical mirror (default True for Prusa)
    """
    img_path = Path(input_path)
    if not img_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    print(f"\n{'═'*68}")
    print(f"  DepthForge Pipeline  –  {img_path.name}")
    print(f"{'═'*68}\n")

    # ── 1. Load image ─────────────────────────────────────────────────────────
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Cannot load image: {input_path}")
    h, w = image.shape[:2]
    print(f"  Input image: {w}×{h} px\n")

    # ── 2. Initialize DepthForge ──────────────────────────────────────────────
    df = DepthForge()
    de_cfg = df.config['model']['depth_estimation']

    midas_compiled = None
    dpt_compiled   = None

    if df.ov_core:
        midas_path = de_cfg.get('midas_model_path')
        if midas_path:
            midas_compiled = df._load_openvino_model(midas_path)
        dpt_path = de_cfg.get('dpt_model_path')
        if dpt_path:
            dpt_compiled = df._load_openvino_model(dpt_path)

    # ── 3. Compute depth maps ─────────────────────────────────────────────────
    print(f"  {'─'*64}")
    print(f"  Computing depth maps…")
    print(f"  {'─'*64}")

    depth_maps = {}
    times      = {}

    def run_method(name: str, fn):
        t0 = time.perf_counter()
        result = fn(image)
        ms = (time.perf_counter() - t0) * 1000
        return name, result, ms

    tasks = [("standard", df.generate_depth_map_midas)]
    if midas_compiled:
        tasks.append(("midas",
                       lambda img, _c=midas_compiled:
                           df._run_openvino_inference(_c, img, "midas")))
    if dpt_compiled:
        tasks.append(("dpt",
                       lambda img, _c=dpt_compiled:
                           df._run_openvino_inference(_c, img, "dpt")))

    # OpenVINO is not thread-safe – sequential execution
    labels_map = {"standard": "Standard (synthetic)",
                  "midas":    "OpenVINO MiDaS v2.1 Small",
                  "dpt":      "OpenVINO DPT Large"}
    for name, fn in tasks:
        name, depth, ms = run_method(name, fn)
        depth_maps[name] = depth
        times[name] = ms
        print(f"  ✓ {labels_map.get(name, name):<30}  {ms:7.1f} ms")

        # Save individual depth map
        cv2.imwrite(str(out_dir / f"{stem}_depth_{name}.png"), depth)
        cv2.imwrite(
            str(out_dir / f"{stem}_depth_{name}_color.jpg"),
            cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    # ── 4. Fusion ─────────────────────────────────────────────────────────────
    print(f"\n  {'─'*64}")
    print(f"  Fusing depth maps (ensemble)…")

    active_weights = {k: ENSEMBLE_WEIGHTS[k] for k in depth_maps}
    total_w = sum(active_weights.values())
    for k, w_val in active_weights.items():
        pct = w_val / total_w * 100
        print(f"    {k:<10}  weight {w_val:.2f}  ({pct:.0f} %)")

    fused_f32  = fuse_depth_maps(depth_maps)
    fused_post = postprocess_depth(fused_f32,
                                   clahe_clip=clahe_clip,
                                   sigma=smooth_sigma)

    # Save ensemble
    cv2.imwrite(str(out_dir / f"{stem}_depth_ensemble.png"), fused_post)
    cv2.imwrite(
        str(out_dir / f"{stem}_depth_ensemble_color.jpg"),
        cv2.applyColorMap(fused_post, cv2.COLORMAP_INFERNO),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    print(f"  ✓ Ensemble saved")

    # ── 5. Comparison grid ────────────────────────────────────────────────────
    _save_comparison(image, depth_maps, fused_post, out_dir, stem)
    print(f"  ✓ Comparison grid saved")

    # ── 6. STL export ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*64}")
    height_mm_est = width_mm * h / w
    print(f"  Generating STL…")
    print(f"    Physical dimensions:  {width_mm:.0f} × {height_mm_est:.0f} × "
          f"{base_mm + relief_mm:.0f} mm")
    print(f"    Mesh resolution: max {mesh_px} px")
    print(f"    invert_depth={invert_depth}  flip_x={flip_x}  flip_y={flip_y}")

    t0 = time.perf_counter()
    stl_path = str(out_dir / f"{stem}_ensemble.stl")
    depth_to_stl(fused_post,
                 stl_path,
                 width_mm     = width_mm,
                 relief_mm    = relief_mm,
                 base_mm      = base_mm,
                 max_px       = mesh_px,
                 invert_depth = invert_depth,
                 flip_x       = flip_x,
                 flip_y       = flip_y)
    t_stl = (time.perf_counter() - t0) * 1000

    print(f"  ✓ STL generated  ({t_stl:.0f} ms)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  OUTPUT FILES  →  {out_dir}")
    print(f"{'═'*68}")
    for f in sorted(out_dir.glob(f"{stem}_*")):
        size = f.stat().st_size
        unit = "KB" if size < 1_048_576 else "MB"
        val  = size / 1024 if size < 1_048_576 else size / 1_048_576
        print(f"  {f.name:<50}  {val:6.1f} {unit}")
    print()


def _save_comparison(orig: np.ndarray, depth_maps: dict,
                     ensemble: np.ndarray,
                     out_dir: Path, stem: str) -> None:
    """Comparison grid 2×N: original + maps + ensemble (JPEG, 2-column layout)."""
    PW = 760
    PH = max(100, int(PW * orig.shape[0] / orig.shape[1]))
    PH = min(PH, 600)

    name_labels = {
        "standard": "Standard synthetic",
        "midas":    "OpenVINO MiDaS v2.1 Small",
        "dpt":      "OpenVINO DPT Large",
    }

    panels_data = [("Original", orig)]
    for key in ["standard", "midas", "dpt"]:
        if key in depth_maps:
            colored = cv2.applyColorMap(depth_maps[key], cv2.COLORMAP_INFERNO)
            panels_data.append((name_labels[key], colored))
    panels_data.append(("Ensemble (fusion)", cv2.applyColorMap(ensemble, cv2.COLORMAP_INFERNO)))

    def make_cell(label, img):
        cell = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(cell, (0, 0), (PW, 36), (15, 15, 15), -1)
        cv2.putText(cell, label, (9, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (255, 255, 255), 1, cv2.LINE_AA)
        return cell

    cells = [make_cell(lbl, img) for lbl, img in panels_data]
    if len(cells) % 2 != 0:
        cells.append(np.zeros((PH, PW, 3), dtype=np.uint8))

    sep_v = np.full((PH, 4, 3), 35, dtype=np.uint8)
    sep_h = np.full((4, PW * 2 + 4, 3), 35, dtype=np.uint8)

    rows = []
    for i in range(0, len(cells), 2):
        rows.append(np.hstack([cells[i], sep_v, cells[i + 1]]))

    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep_h, row])

    out_path = out_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DepthForge – depth pipeline + STL export for 3D printing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to the input image")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Output directory")
    parser.add_argument("--width-mm",  type=float, default=200.0,
                        help="STL model width [mm]")
    parser.add_argument("--relief-mm", type=float, default=10.0,
                        help="Max relief height [mm]")
    parser.add_argument("--base-mm",   type=float, default=3.0,
                        help="Base plate thickness [mm]")
    parser.add_argument("--mesh-px",   type=int,   default=512,
                        help="Max STL mesh resolution [px]")
    parser.add_argument("--clahe-clip", type=float, default=2.5,
                        help="CLAHE clip limit")
    parser.add_argument("--smooth-sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma [px]")
    parser.add_argument("--invert-depth", action="store_true", default=False,
                        help="Invert depth values (convex↔concave)")
    parser.add_argument("--flip-x", action="store_true", default=False,
                        help="Horizontal mirror of STL")
    parser.add_argument("--no-flip-y", action="store_true", default=False,
                        help="Disable default vertical flip (flip_y=True corrects "
                             "image→Prusa Slicer orientation)")

    args = parser.parse_args()

    run_pipeline(
        input_path    = args.input,
        output_dir    = args.output_dir,
        width_mm      = args.width_mm,
        relief_mm     = args.relief_mm,
        base_mm       = args.base_mm,
        mesh_px       = args.mesh_px,
        clahe_clip    = args.clahe_clip,
        smooth_sigma  = args.smooth_sigma,
        invert_depth  = args.invert_depth,
        flip_x        = args.flip_x,
        flip_y        = not args.no_flip_y,
    )


if __name__ == "__main__":
    main()

