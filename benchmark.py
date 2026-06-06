#!/usr/bin/env python3
"""
DepthForge – benchmark comparing depth map generation methods:
  • Standard (synthetic, no AI model)
  • OpenVINO MiDaS v2.1 Small
  • OpenVINO DPT Large
  • Ensemble (weighted fusion of all available methods)

Results are saved to the output/benchmark_<timestamp>/ directory.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent / "src"))
from depth_forge import DepthForge

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGES = [
    "data/sample_input.jpg",
    "data/museum_sample_input.jpg",
    "data/Jozef_Chełmonski_-_Indian_summer_-_Google_Art_Project.jpg",
    "data/Stańczyk.jpg",
]
RUNS = 3  # how many times to run each method (for reliable timing)

# Ensemble weights: DPT (highest quality) > MiDaS > Standard (synthetic)
ENSEMBLE_WEIGHTS = {
    "Standard (synthetic)":      0.15,
    "OpenVINO MiDaS v2.1 Small": 0.35,
    "OpenVINO DPT Large":        0.50,
}

OUTPUT_BASE = Path("output") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Terminal colours ──────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
MAGENTA= "\033[35m"
RED    = "\033[31m"
RESET  = "\033[0m"


def separator(char="─", width=72):
    print(char * width)


def print_header(text):
    separator("═")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    separator("═")


def timeit(fn, *args, runs=RUNS):
    """Runs fn(*args) `runs` times, returns (last_result, avg_ms, min_ms)."""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t0) * 1000)
    return result, sum(times) / len(times), min(times)


# ── Ensemble helpers ──────────────────────────────────────────────────────────

def _normalize_f32(arr: np.ndarray) -> np.ndarray:
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr.astype(np.float32) - a_min) / (a_max - a_min)


def build_ensemble(depth_results: dict) -> np.ndarray:
    """
    Weighted fusion of available depth maps followed by CLAHE post-processing.
    Uses ENSEMBLE_WEIGHTS; re-normalises weights to available methods.
    """
    acc = None
    weight_sum = 0.0

    for method_name, depth in depth_results.items():
        w = ENSEMBLE_WEIGHTS.get(method_name, 0.0)
        if w == 0.0 or depth is None:
            continue
        norm = _normalize_f32(depth)
        acc = norm * w if acc is None else acc + norm * w
        weight_sum += w

    if acc is None or weight_sum < 1e-9:
        return None

    fused = (acc / weight_sum).astype(np.float32)

    # Gaussian smoothing + CLAHE
    smoothed = gaussian_filter(fused, sigma=1.5)
    img_u8   = (_normalize_f32(smoothed) * 255).astype(np.uint8)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img_u8)


# ── Comparison grid ───────────────────────────────────────────────────────────

def save_comparison_grid(image_path: str, results: dict,
                         ensemble: np.ndarray, out_dir: Path):
    """
    Saves a comparison grid (JPEG) with:
      original | standard | midas | dpt | ensemble
    laid out in a 2-column layout.
    """
    orig = cv2.imread(image_path)
    if orig is None:
        return None

    PW = 800
    PH = min(600, int(PW * orig.shape[0] / orig.shape[1]))

    panels = [("Original", orig)]
    for method_name, depth in results.items():
        panels.append((method_name, cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)))
    if ensemble is not None:
        panels.append((f"Ensemble ({len(results)} methods)",
                       cv2.applyColorMap(ensemble, cv2.COLORMAP_INFERNO)))

    def make_cell(label, img):
        cell = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(cell, (0, 0), (PW, 34), (20, 20, 20), -1)
        cv2.putText(cell, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)
        return cell

    cells = [make_cell(lbl, img) for lbl, img in panels]
    if len(cells) % 2 != 0:
        cells.append(np.zeros((PH, PW, 3), dtype=np.uint8))

    sep_v = np.full((PH, 4, 3), 40, dtype=np.uint8)
    sep_h = np.full((4, PW * 2 + 4, 3), 40, dtype=np.uint8)

    rows = []
    for i in range(0, len(cells), 2):
        rows.append(np.hstack([cells[i], sep_v, cells[i + 1]]))

    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, sep_h, row])

    stem     = Path(image_path).stem
    out_path = out_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark():
    print_header("DepthForge – Depth Method Benchmark  (all methods + ensemble)")

    df = DepthForge()

    # ── Define methods under test ─────────────────────────────────────────────
    methods = {}

    # 1. Standard method (synthetic)
    methods["Standard (synthetic)"] = df.generate_depth_map_midas

    # 2. OpenVINO MiDaS
    midas_path     = df.config["model"]["depth_estimation"].get("midas_model_path")
    midas_compiled = df._load_openvino_model(midas_path) if (df.ov_core and midas_path) else None
    if midas_compiled:
        methods["OpenVINO MiDaS v2.1 Small"] = \
            lambda img, _c=midas_compiled: df._run_openvino_inference(_c, img, "midas")
    else:
        print(f"{YELLOW}  ⚠ MiDaS OpenVINO unavailable{RESET}")

    # 3. OpenVINO DPT Large
    dpt_path     = df.config["model"]["depth_estimation"].get("dpt_model_path")
    dpt_compiled = df._load_openvino_model(dpt_path) if (df.ov_core and dpt_path) else None
    if dpt_compiled:
        methods["OpenVINO DPT Large"] = \
            lambda img, _c=dpt_compiled: df._run_openvino_inference(_c, img, "dpt")
    else:
        print(f"{YELLOW}  ⚠ DPT OpenVINO unavailable{RESET}")

    # Print active ensemble weights
    active_w   = {m: ENSEMBLE_WEIGHTS[m] for m in methods if m in ENSEMBLE_WEIGHTS}
    total_w    = sum(active_w.values())
    weight_str = "  |  ".join(
        f"{m}: {w/total_w*100:.0f}%" for m, w in active_w.items()
    )

    print(f"  Methods tested   : {BOLD}{', '.join(methods)}{RESET}")
    print(f"  Ensemble weights : {weight_str}")
    print(f"  Test images      : {len([p for p in IMAGES if Path(p).exists()])}/{len(IMAGES)}")
    print(f"  Repetitions/image: {RUNS}")
    print()

    # ── Main loop ─────────────────────────────────────────────────────────────
    global_results = {}   # { method_name: [avg_ms, ...] }
    ensemble_times = []

    for img_path in IMAGES:
        if not Path(img_path).exists():
            print(f"{YELLOW}  Skipped (file not found): {img_path}{RESET}\n")
            continue

        image = cv2.imread(img_path)
        h, w  = image.shape[:2]
        stem  = Path(img_path).stem

        separator()
        print(f"{BOLD}  Image: {img_path}  ({w}×{h} px){RESET}")
        separator()

        depth_results = {}

        for method_name, fn in methods.items():
            depth, avg_ms, min_ms = timeit(fn, image)
            depth_results[method_name] = depth

            safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
            out_file  = OUTPUT_BASE / f"{stem}_{safe_name}.png"
            cv2.imwrite(str(out_file), depth)

            # Colour version (INFERNO)
            colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(OUTPUT_BASE / f"{stem}_{safe_name}_color.png"), colored)

            status = f"{GREEN}✓{RESET}"
            print(f"  {status} {BOLD}{method_name:<30}{RESET}"
                  f"  avg: {CYAN}{avg_ms:7.1f} ms{RESET}"
                  f"  min: {min_ms:7.1f} ms"
                  f"  → {out_file.name}")

            global_results.setdefault(method_name, []).append(avg_ms)

        # ── Ensemble ──────────────────────────────────────────────────────────
        t0       = time.perf_counter()
        ensemble = build_ensemble(depth_results)
        ens_ms   = (time.perf_counter() - t0) * 1000
        ensemble_times.append(ens_ms)

        if ensemble is not None:
            ens_file = OUTPUT_BASE / f"{stem}_ensemble.png"
            cv2.imwrite(str(ens_file), ensemble)
            ens_color_file = OUTPUT_BASE / f"{stem}_ensemble_color.png"
            cv2.imwrite(str(ens_color_file),
                        cv2.applyColorMap(ensemble, cv2.COLORMAP_INFERNO))
            print(f"  {MAGENTA}⊕ Ensemble ({len(depth_results)} methods){' '*13}{RESET}"
                  f"  fuse:  {MAGENTA}{ens_ms:7.1f} ms{RESET}"
                  f"            → {ens_file.name}")

        # ── Comparison grid ───────────────────────────────────────────────────
        grid_path = save_comparison_grid(img_path, depth_results, ensemble, OUTPUT_BASE)
        if grid_path:
            print(f"  {'':31}  Grid: {grid_path.name}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    separator("═")
    print(f"{BOLD}{CYAN}  SUMMARY – average time per image{RESET}")
    separator("═")

    method_avgs = {m: sum(ts) / len(ts) for m, ts in global_results.items()}
    if ensemble_times:
        method_avgs["⊕ Ensemble (fusion)"] = sum(ensemble_times) / len(ensemble_times)

    baseline = method_avgs.get("Standard (synthetic)", None)
    rows     = sorted(method_avgs.items(), key=lambda x: x[1])

    for method_name, avg_ms in rows:
        speedup = ""
        if baseline and method_name not in ("Standard (synthetic)", "⊕ Ensemble (fusion)"):
            ratio   = avg_ms / baseline
            speedup = f"  ({ratio:.1f}× slower than synthetic)"
        is_fastest = method_name == rows[0][0]
        is_ensemble = method_name.startswith("⊕")
        marker = (f"{GREEN}★ FASTEST{RESET}      " if is_fastest
                  else f"{MAGENTA}⊕ ENSEMBLE{RESET}     " if is_ensemble
                  else "             ")
        color  = MAGENTA if is_ensemble else CYAN
        print(f"  {marker}{BOLD}{method_name:<35}{RESET}  {color}{avg_ms:7.1f} ms{RESET}{speedup}")

    separator("═")
    print(f"\n  Results saved in: {BOLD}{OUTPUT_BASE}{RESET}\n")

    # ── List output files ──────────────────────────────────────────────────────
    all_files = sorted(OUTPUT_BASE.glob("*"))
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"  Output files ({len(all_files)} files, "
          f"{total_size/1024/1024:.1f} MB total):")
    for f in all_files:
        size = f.stat().st_size
        unit = "KB" if size < 1_048_576 else "MB"
        val  = size / 1024 if size < 1_048_576 else size / 1_048_576
        print(f"    {f.name:<60}  {val:6.1f} {unit}")
    print()


if __name__ == "__main__":
    run_benchmark()
