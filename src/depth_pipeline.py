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
from scipy.ndimage import gaussian_filter, median_filter
from stl import mesh as stl_mesh
from stl.stl import Mode as StlMode

# ── Path to the DepthForge module ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from depth_forge import DepthForge

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH MAP FUSION  –  scale-shift alignment (MiDaS benchmark standard)
# ─────────────────────────────────────────────────────────────────────────────

# Reference model: best quality first.  All other maps are aligned TO it.
# Weights are applied AFTER alignment (maps are already in the same scale).
ENSEMBLE_WEIGHTS = {
    "dpt":      0.50,
    "midas":    0.35,
    "standard": 0.15,
}

# Preference order for choosing the reference map
_REFERENCE_PRIORITY = ["dpt", "midas", "standard"]


def normalize_f32(arr: np.ndarray) -> np.ndarray:
    """Normalizes a 2D array to the range [0, 1] float32 (hard min-max)."""
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr.astype(np.float32) - a_min) / (a_max - a_min))


def normalize_f32_robust(arr: np.ndarray,
                         low_pct: float = 1.0,
                         high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-based normalization to [0, 1] float32.

    Clips outliers at the given percentiles before stretching, which prevents
    a handful of extreme pixels (model noise, image borders) from squashing
    the rest of the tonal range into a narrow band – the root cause of
    'blown-out' edges and flat, detail-less areas.

    Args:
        arr:      Input array (any float dtype).
        low_pct:  Lower percentile for clipping (default 1.0).
        high_pct: Upper percentile for clipping (default 99.0).

    Returns:
        float32 array in [0, 1].
    """
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def align_scale_shift(source: np.ndarray,
                      target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Aligns 'source' to 'target' by least-squares fitting  y = s·x + t.

    This is the standard MiDaS benchmark alignment: because MiDaS/DPT output
    inverse depth with an unknown, model-specific scale and shift, direct
    averaging of [0-1]-normalised maps is wrong – it conflates two unknowns
    (scale + offset) into one normalisation step.  Least-squares solves both.

    Both maps must be in the SAME polarity convention before calling this
    function (see fix_polarity).

    Args:
        source: map to transform (any float range)
        target: reference map    (any float range)

    Returns:
        (aligned_source, scale s, shift t)
    """
    x = source.flatten().astype(np.float64)
    y = target.flatten().astype(np.float64)

    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    s, t = result[0]

    aligned = (s * source + t).astype(np.float32)
    return aligned, float(s), float(t)


def fix_polarity(depth_map: np.ndarray,
                 reference_map: np.ndarray) -> np.ndarray:
    """
    Checks Pearson correlation between maps; if negative, inverts depth_map.

    A negative correlation means one map uses disparity (bright = near) while
    the other uses metric depth (bright = far).  The MiDaS family uses
    inverse depth (disparity), but the Standard synthetic model may differ.

    Args:
        depth_map:     map whose polarity may need fixing
        reference_map: trusted reference (e.g. MiDaS or DPT)

    Returns:
        depth_map with corrected polarity (float32)
    """
    depth_map = depth_map.astype(np.float32)
    corr = float(np.corrcoef(depth_map.flatten(),
                             reference_map.flatten())[0, 1])
    if corr < 0:
        logger.debug("fix_polarity: negative correlation (%.3f) – inverting map", corr)
        depth_map = depth_map.max() - depth_map
    return depth_map


def fuse_depth_maps(depth_maps_f32: dict) -> np.ndarray:
    """
    Fuses depth maps using MiDaS-standard scale-shift (least-squares) alignment.

    Pipeline:
      1. Choose reference (highest-priority available model).
      2. Fix polarity of every non-reference map vs. the reference.
      3. Align each non-reference map to the reference via least-squares.
      4. Weighted average with confidence masking (reject top-10 % discrepancy).
      5. Self-guided filter for edge-preserving smoothing.

    Args:
        depth_maps_f32: {'standard': ndarray, 'midas': ndarray, 'dpt': ndarray}
                        Values should be float32 (raw model output preferred).

    Returns:
        Fused depth map float32 [0, 1]
    """
    available = {k: v.astype(np.float32)
                 for k, v in depth_maps_f32.items()
                 if v is not None}
    if not available:
        raise ValueError("No depth maps available for fusion.")

    # ── 1. Pick reference ────────────────────────────────────────────────────
    ref_key = next((k for k in _REFERENCE_PRIORITY if k in available), None)
    if ref_key is None:
        ref_key = next(iter(available))
    reference = available[ref_key]
    logger.info("Fusion reference model: %s", ref_key)

    # ── 2 & 3. Fix polarity + align every map to reference ──────────────────
    aligned: dict[str, np.ndarray] = {}
    for key, depth in available.items():
        if key == ref_key:
            aligned[key] = reference
            continue
        fixed = fix_polarity(depth, reference)
        aligned_map, s, t = align_scale_shift(fixed, reference)
        logger.debug("align %s→%s: s=%.4f  t=%.4f", key, ref_key, s, t)
        aligned[key] = aligned_map

    # ── 4. Weighted average with SOFT Gaussian confidence ────────────────────
    # Instead of hard-switching pixels above the 90th-percentile discrepancy
    # to the reference value (which creates flat, detail-less patches in
    # fabric folds and faces), we use a smooth Gaussian weight that falls off
    # gracefully as local disagreement grows.  Areas where models agree get
    # full weight; areas with moderate disagreement are blended; only extreme
    # outliers are down-weighted – without ever zeroing out local structure.
    weight_sum = 0.0
    acc        = np.zeros_like(reference, dtype=np.float64)
    counts     = np.zeros_like(reference, dtype=np.float64)

    for key, depth in aligned.items():
        w = ENSEMBLE_WEIGHTS.get(key, 0.0)
        if w == 0.0:
            continue
        diff      = np.abs(depth - reference).astype(np.float64)
        sigma_conf = float(np.std(diff)) + 1e-6
        # Gaussian confidence: 1 where diff≈0, falls to ~0.14 at diff=2σ
        confidence = np.exp(-(diff ** 2) / (2.0 * sigma_conf ** 2))

        w_eff  = w * confidence
        acc    += w_eff * depth.astype(np.float64)
        counts += w_eff
        weight_sum += w

    if weight_sum < 1e-9:
        raise ValueError("All ensemble weights are zero.")

    fused = (acc / np.clip(counts, 1e-6, None)).astype(np.float32)

    # ── 5. Edge-preserving smoothing (guided filter) ─────────────────────────
    fused_norm = normalize_f32_robust(fused)   # percentile-based [0, 1] float32
    try:
        # Use the fused depth map itself as guide.
        # Using the RGB image as guide transfers paint brushstroke edges into
        # the depth map (false discontinuities).  Self-guided filtering
        # smooths noise while preserving genuine depth transitions only.
        guide_u8 = (fused_norm * 255).astype(np.uint8)

        fused_smooth = cv2.ximgproc.guidedFilter(
            guide=guide_u8,
            src=fused_norm,
            radius=4,
            eps=0.08,   # higher ε → more smoothing, fewer artefacts in paintings
        )
        logger.info("Guided filter applied (self-guided, radius=4, eps=0.08)")
        return normalize_f32_robust(fused_smooth)
    except AttributeError:
        from scipy.ndimage import gaussian_filter as _gf
        logger.warning("cv2.ximgproc not available – falling back to Gaussian σ=1.5")
        return normalize_f32_robust(_gf(fused_norm, sigma=1.5))


def postprocess_depth(depth_f32: np.ndarray,
                      clahe_clip: float = 2.0,
                      sigma: float = 0.7,
                      gamma: float = 1.02) -> np.ndarray:
    """
    Enhance the fused depth map for natural, print-ready output:
      1. Gaussian smoothing  – light final de-noising (guided filter already
                               handles global smoothing; keep sigma low to
                               preserve fabric folds and facial details)
      2. Mild CLAHE          – small tiles, moderate clip → local contrast
                               without tile-boundary banding
      3. Gamma correction    – minimal tone compression of bright areas

    Args:
        depth_f32:  map [0, 1] float32 (output of fuse_depth_maps)
        clahe_clip: CLAHE contrast limit (default 2.0 – recovers fine detail
                    without introducing aggressive noise)
        sigma:      Gaussian σ in pixels (default 0.7 – guided filter already
                    smooths globally; heavy sigma here destroys cloth/face detail)
        gamma:      Gamma for tone-mapping bright areas (default 1.02 – nearly
                    neutral; avoids compressing highlights into flat plateau)

    Returns:
        Processed map uint8 [0, 255]
    """
    # ── 1. Gaussian – light final pass; guided filter already smoothed globally
    smoothed = gaussian_filter(depth_f32, sigma=sigma) if sigma > 0 else depth_f32

    # ── 2. Convert to uint8 (percentile-robust stretch) ───────────────────────
    img_u8 = (normalize_f32_robust(smoothed) * 255).astype(np.uint8)

    # ── 3. CLAHE – smaller 8×8 tiles = more local contrast; clip 2.0 recovers
    #              fine detail in folds and faces without amplifying noise ──────
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        img_u8 = clahe.apply(img_u8)

    # ── 4. Gamma correction – very light, avoids flattening bright foreground ──
    if abs(gamma - 1.0) > 1e-3:
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
        img_u8 = lut.astype(np.uint8)[img_u8]

    return img_u8


# ─────────────────────────────────────────────────────────────────────────────
#  MICRO-DETAIL OVERLAY  –  high-frequency texture from original image
# ─────────────────────────────────────────────────────────────────────────────

def extract_micro_detail(original_bgr: np.ndarray,
                         strength: float = 0.15,
                         blur_sigma: float = 1.2) -> np.ndarray:
    """
    Extracts high-frequency micro-detail (fabric folds, skin texture, ground)
    from the original image using an unsharp / high-pass method.

    The ensemble depth models (DPT, MiDaS) are trained to predict smooth,
    global scene geometry and never encode surface micro-texture as depth
    differences.  This function recovers that missing layer directly from
    image luminance so it can be additively composited onto the geometry map.

    Args:
        original_bgr: Source image in BGR uint8.
        strength:     Amplitude of the detail layer relative to the [0,1]
                      depth range.  0.10–0.20 is typical; higher values add
                      noise, lower values produce flat surfaces.
        blur_sigma:   Low-pass cutoff (px).  Controls what counts as
                      "texture" vs "geometry":
                        • 1.0–1.2 → fine detail (skin pores, canvas grain)
                        • 1.5–2.0 → coarser detail (broad folds, brushstrokes)
                      The actual Gaussian kernel uses 8× this value so that
                      the cut-off frequency is well below the Nyquist limit.

    Returns:
        float32 array in ≈ [-0.3, 0.3] (clamped), same spatial size as input.
        Values are NOT normalised to [0,1] – they are additive offsets.
    """
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Low-pass: broad Gaussian captures global luminance variations (shading,
    # lighting gradients) that correlate with geometry, not texture.
    low_freq = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma * 8)
    # High-pass residual = original − low_freq = texture only
    high_freq = gray - low_freq
    # Normalise contrast so that the amplitude is scene-independent
    high_freq = high_freq / (float(np.std(high_freq)) + 1e-6)
    # Scale and clip: max ±0.3 means the detail layer contributes at most
    # 30 % of the full relief height, preventing print artefacts from noise
    return np.clip(high_freq * strength, -0.3, 0.3).astype(np.float32)


def apply_detail_overlay(fused_depth_f32: np.ndarray,
                         original_bgr: np.ndarray,
                         strength: float = 0.15,
                         blur_sigma: float = 1.2) -> np.ndarray:
    """
    Composites a micro-detail layer onto the fused depth map.

    This separates two concerns:
      • fused_depth_f32 – coarse, accurate scene geometry (from ensemble)
      • detail layer     – surface micro-texture (from image luminance)

    The detail layer is bicubically resampled to match the depth map
    resolution before addition, so mismatched input sizes are handled
    transparently.

    Args:
        fused_depth_f32: Fused depth map [0, 1] float32.
        original_bgr:    Original source image (BGR uint8).
        strength:        Passed to extract_micro_detail (default 0.15).
        blur_sigma:      Passed to extract_micro_detail (default 1.2).

    Returns:
        float32 depth map [0, 1] with micro-texture embedded.
    """
    detail = extract_micro_detail(original_bgr, strength=strength,
                                  blur_sigma=blur_sigma)
    if detail.shape != fused_depth_f32.shape:
        detail = cv2.resize(detail,
                            (fused_depth_f32.shape[1], fused_depth_f32.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
    combined = fused_depth_f32 + detail
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  TACTILE / TYFLOGRAPHIC PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def fill_small_object_holes(fused_depth_f32: np.ndarray,
                             min_area:    int = 20,
                             max_area:    int = 2000,
                             kernel_size: int = 5) -> np.ndarray:
    """
    Wypełnia puste wnętrza małych, izolowanych obiektów (np. pies w tle),
    które mają wykryty kontur ale płaskie/puste wnętrze z powodu zbyt małej
    rozdzielczości dla modeli depth estimation.

    Modele DPT/MiDaS przy małych obiektach (kilkadziesiąt px) wykrywają tylko
    krawędź sylwetki jako przejście jasność/ciemność w luminancji obrazu, ale
    wnętrze jednolitej ciemnej plamy (np. czarny pies) pozostaje na poziomie
    głębi zbliżonym do tła — tworząc efekt "obrysowany, ale pusty w środku".

    Algorytm:
      1. Canny na mapie głębi → szkielet krawędzi
      2. MORPH_CLOSE → zamknięte kontury (pierścienie)
      3. findContours → lista małych obiektów w zakresie [min_area, max_area] px²
      4. Dla każdego obiektu: wypełnij wnętrze wartością 75. percentyla pikseli
         brzegowych (zamiast środkowej wartości tła)

    Wywołaj PO fuzji ensemble, PRZED detail_overlay i prepare_for_touch, żeby
    wypełniony obszar przeszedł przez cały dalszy łańcuch wygładzania.

    Args:
        fused_depth_f32: Mapa głębi [0, 1] float32.
        min_area:        Minimalna powierzchnia konturu [px²] (filtr szumu).
        max_area:        Maksymalna powierzchnia konturu [px²] (pomija duże bryły).
        kernel_size:     Rozmiar jądra morfologicznego do zamykania konturów.

    Returns:
        float32 mapa głębi [0, 1] z wypełnionymi małymi obiektami.
    """
    depth_u8 = (fused_depth_f32 * 255).astype(np.uint8)
    edges    = cv2.Canny(depth_u8, 30, 90)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (kernel_size, kernel_size))
    closed   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    result = fused_depth_f32.copy()

    filled = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Maska wypełnionego wnętrza konturu
            mask = np.zeros(fused_depth_f32.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            # Maska samego brzegu (gradient morfologiczny)
            border_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0
            if border_mask.any():
                # 75. percentyl pikseli brzegowych → "wypukłość" wypełnienia
                fill_value = float(np.percentile(fused_depth_f32[border_mask], 75))
                # Podnosimy tylko te piksele wnętrza, które są niżej od wartości brzegu
                interior = mask > 0
                result[interior] = np.maximum(result[interior], fill_value)
                filled += 1

    if filled:
        logger.info("fill_small_object_holes: filled %d objects "
                    "(area %d–%d px²)", filled, min_area, max_area)
    return result.astype(np.float32)


def prepare_for_touch(fused_depth_f32: np.ndarray,
                      median_size: int   = 5,
                      sigma:       float = 3.5) -> np.ndarray:
    """
    Prepares a depth map for tactile / tyflographic 3D printing.

    The touch sense has very different requirements from vision:
      • Resolves smooth, large-scale height changes well (body contours, hands)
      • Cannot distinguish spatial noise below ~1-2 mm – interprets it as
        unpleasant, sharp roughness rather than meaningful texture
      • Benefits from a small number of clearly distinct height levels

    The two-stage smoothing pipeline addresses this:
      1. Median filter (default size=5) — removes isolated 'spike' outliers
         (single-pixel anomalies from the ensemble) without blurring the main
         depth contours.  Must come BEFORE Gaussian so the Gauss works on
         clean data.
      2. Gaussian (default σ=3.5) — smooths everything to a touch-friendly
         surface, eliminating the thousands of micro-jumps that are invisible
         on screen but physically felt as coarse texture on the print.

    Args:
        fused_depth_f32: Depth map [0, 1] float32 (output of fuse_depth_maps).
        median_size:     Footprint size for scipy.ndimage.median_filter.
                         Odd integers only; 5 is a good starting point.
        sigma:           Gaussian σ [px] for the final smoothing pass.
                         3.0–4.0 gives a comfortable tactile surface.

    Returns:
        float32 depth map [0, 1], smooth and spike-free.
    """
    # Step 1: spike removal – median_filter on float preserves range [0,1]
    cleaned = median_filter(fused_depth_f32.astype(np.float64),
                            size=median_size).astype(np.float32)
    # Step 2: broad Gaussian – produces the smooth, "palm-readable" relief
    smoothed = gaussian_filter(cleaned, sigma=sigma)
    # Re-normalise so the full [0,1] range is used after smoothing
    return normalize_f32_robust(smoothed)


def prepare_for_touch_multiscale(fused_depth_f32: np.ndarray,
                                 median_size:   int   = 5,
                                 fine_sigma:    float = 1.5,
                                 limb_sigma:    float = 3.0) -> np.ndarray:
    """
    Wieloskalowe wygładzanie mapy głębi dla druku taktylnego.

    Dotychczasowe podejście (jeden Gauss σ=3.5) traktuje fakturę tkaniny i
    kontury kończyn identycznie — albo usuwa oba (duże σ), albo zachowuje oba
    (małe σ). Rozróżnienie tych dwóch skal przestrzennych jest kluczem do
    zachowania czytelnego układu nóg/rąk przy jednoczesnym usunięciu
    kłujących mikrodetali.

    Trzy pasma częstotliwości:
      • fine  (< fine_sigma px):           faktura tkaniny/trawy → USUWANA
      • limb  (fine_sigma … limb_sigma):   kontury kończyn, dłoni → ZACHOWANA
      • coarse (> limb_sigma):             globalna sylwetka → zawsze gładka

    Algorytm:
      1. Mediana – usunięcie pojedynczych „szpilkowych" outlierów.
      2. Gauss σ=fine_sigma – usuwa wyłącznie sub-pikselowy szum tekstury;
         za mały, by dotknąć konturów kończyn (rzędu dziesiątek pikseli).
      3. Gauss σ=limb_sigma*0.5 – finalne, delikatne domknięcie krawędzi
         kończyn; promień celowo mniejszy niż limb_sigma, żeby nie zlewać
         blisko siebie leżących nóg w jedną bryłę.

    Args:
        fused_depth_f32: Mapa głębi [0, 1] float32.
        median_size:     Rozmiar filtra medianowego (spike removal, domyślnie 5).
        fine_sigma:      σ [px] dla usuwania sub-pikselowej tekstury (1.2–1.5).
        limb_sigma:      σ [px] definiujący skalę kończyn; filtr końcowy używa
                         limb_sigma*0.5, żeby zachować separację nóg (2.5–3.5).

    Returns:
        float32 mapa głębi [0, 1] z zachowanymi konturami kończyn i usuniętą
        drobną teksturą.
    """
    # Krok 1: spike removal – mediana na float64 zachowuje zakres [0,1]
    cleaned = median_filter(fused_depth_f32.astype(np.float64),
                            size=median_size).astype(np.float32)

    # Krok 2: usuń wyłącznie sub-pikselowy szum tekstury (< fine_sigma)
    denoised = gaussian_filter(cleaned, sigma=fine_sigma)

    # Krok 3: delikatne domknięcie konturów kończyn; promień = limb_sigma*0.5
    # zapobiega zlaniu się blisko siebie leżących nóg/rąk w jedną bryłę
    limb_preserving = gaussian_filter(denoised, sigma=limb_sigma * 0.5)

    return normalize_f32_robust(limb_preserving)


def quantize_depth(depth_f32: np.ndarray,
                   levels: int = 5) -> np.ndarray:
    """
    Quantizes a continuous depth map to a fixed number of discrete height
    levels, following museum guidelines for tactile graphics.

    Best-practice tyflographic standards (e.g. Muzeum Prado, RNIB) recommend
    3–5 clearly distinct height levels per object rather than a continuous
    gradient, because fingers distinguish steps better than gradients.

    Equal-area quantization is used (np.percentile boundaries) so that each
    level occupies roughly the same area on the print, avoiding one level
    dominating the surface.

    Args:
        depth_f32: Depth map [0, 1] float32.
        levels:    Number of discrete height steps (3–6 recommended for touch).

    Returns:
        float32 depth map [0, 1] with exactly `levels` distinct values.
        Each step is spaced uniformly: 0, 1/(levels-1), …, 1.
    """
    if levels < 2:
        return depth_f32.copy()

    # Compute equal-area percentile boundaries (histogram equalization on levels)
    boundaries = np.percentile(depth_f32,
                               np.linspace(0, 100, levels + 1))
    # Assign each pixel to a level index via np.digitize
    level_idx = np.digitize(depth_f32, boundaries[1:-1])  # 0 … levels-1
    # Map level index to evenly spaced float values
    step = 1.0 / (levels - 1)
    quantized = (level_idx * step).astype(np.float32)
    return np.clip(quantized, 0.0, 1.0)


def quantize_depth_foreground_aware(depth_f32: np.ndarray,
                                    fg_threshold_pct: float = 40.0,
                                    bg_levels: int = 2,
                                    fg_levels: int = 4,
                                    ) -> tuple:
    """
    Kwantyzuje mapę głębi z asymetrycznym podziałem poziomów: mniej poziomów
    dla tła (niebo/ziemia), więcej dla pierwszego planu (postać).

    Standardowa kwantyzacja equal-area oblicza granice na CAŁEJ mapie, więc
    rozległe tło (niebo, podłoga) pochłania większość poziomów, a główny
    obiekt sceny (postać, twarz) dostaje 1–2 poziomy – praktycznie
    nierozróżnialne dotykiem.  Ten wariant dzieli zakres głębi na dwie strefy
    (tło / pierwszy plan) i przydziela im oddzielne pule poziomów.

    Args:
        depth_f32:         Mapa głębi [0, 1] float32.
        fg_threshold_pct:  Percentyl odcinający tło od pierwszego planu
                           (domyślnie 40 – dolne 40 % wartości = tło/daleko).
        bg_levels:         Liczba dyskretnych poziomów dla tła (domyślnie 2).
        fg_levels:         Liczba dyskretnych poziomów dla pierwszego planu
                           (domyślnie 4).

    Returns:
        Tuple (quantized_f32, level_idx, n_levels):
          • quantized_f32 – float32 [0, 1] z (bg_levels + fg_levels) poziomami.
          • level_idx     – int32 tablica 0…(n_levels-1) – INDEKSY całkowite,
                            wolne od błędów zaokrąglenia float; wymagane przez
                            smooth_quantized_boundaries().
          • n_levels      – łączna liczba poziomów (bg_levels + fg_levels).
    """
    n_levels = bg_levels + fg_levels
    fg_cutoff = float(np.percentile(depth_f32, fg_threshold_pct))
    bg_mask = depth_f32 <= fg_cutoff
    fg_mask = ~bg_mask

    # Mapa indeksów całkowitych 0…(n_levels-1); tło zajmuje 0…(bg_levels-1),
    # pierwszy plan zajmuje bg_levels…(n_levels-1).
    level_idx = np.zeros(depth_f32.shape, dtype=np.int32)
    result    = np.zeros_like(depth_f32)

    # ── Tło: niskie wartości głębi (daleko/niebo) ─────────────────────────────
    if bg_mask.any() and bg_levels >= 2:
        bg_bounds = np.percentile(depth_f32[bg_mask],
                                  np.linspace(0, 100, bg_levels + 1))
        # Indeks całkowity w zakresie 0…(bg_levels-1) dla każdego piksela
        raw_bg_idx = np.digitize(depth_f32, bg_bounds[1:-1])      # 0…bg_levels-1
        # Zabezpieczenie: clip na wypadek pikseli dokładnie na granicy percentyla
        raw_bg_idx = np.clip(raw_bg_idx, 0, bg_levels - 1)
        level_idx[bg_mask] = raw_bg_idx[bg_mask]
        # Wartość float: równomiernie w [0, fg_cutoff]
        result[bg_mask] = (raw_bg_idx[bg_mask].astype(np.float32)
                           / max(bg_levels - 1, 1)) * fg_cutoff
    elif bg_mask.any():
        result[bg_mask]    = depth_f32[bg_mask]
        level_idx[bg_mask] = 0

    # ── Pierwszy plan: wysokie wartości głębi (blisko/postać) ─────────────────
    if fg_mask.any() and fg_levels >= 2:
        fg_bounds = np.percentile(depth_f32[fg_mask],
                                  np.linspace(0, 100, fg_levels + 1))
        raw_fg_idx = np.digitize(depth_f32, fg_bounds[1:-1])      # 0…fg_levels-1
        raw_fg_idx = np.clip(raw_fg_idx, 0, fg_levels - 1)
        # Przesuń indeks o bg_levels, żeby fg zajmował bg_levels…(n_levels-1)
        level_idx[fg_mask] = raw_fg_idx[fg_mask] + bg_levels
        fg_range = 1.0 - fg_cutoff
        result[fg_mask] = fg_cutoff + (raw_fg_idx[fg_mask].astype(np.float32)
                                       / max(fg_levels - 1, 1)) * fg_range
    elif fg_mask.any():
        result[fg_mask]    = depth_f32[fg_mask]
        level_idx[fg_mask] = bg_levels

    return np.clip(result, 0.0, 1.0).astype(np.float32), level_idx, n_levels


def smooth_quantized_boundaries(quantized_f32: np.ndarray,
                                level_idx: np.ndarray,
                                n_levels: int,
                                kernel_size: int = 9) -> np.ndarray:
    """
    Usuwa postrzępione krawędzie między poziomami kwantyzacji przez morfologiczne
    domknięcie i otwarcie każdej maski poziomu osobno, a następnie ponowne złożenie.

    Nawet po wygładzeniu gaussowskim mapa głębi zachowuje resztkowy szum, który
    powoduje, że piksele przy granicy progów wielokrotnie przeskakują między
    poziomami – efekt „staircase noise".  Na wydruku dotykowym odczuwany jest
    jako fizyczna „piłka" wzdłuż krawędzi sylwetki.  Morfologia na maskach
    poziomów eliminuje ten efekt bez zmiany globalnej geometrii.

    WAŻNE: funkcja identyfikuje poziomy po INDEKSIE całkowitym (level_idx),
    NIE po wartości float quantized_f32.  Identyfikacja po float jest podatna
    na błędy zaokrągleń – te same „logiczne" poziomy mogą mieć dziesiątki
    lekko różniących się wartości float w różnych miejscach mapy, co powoduje
    przetwarzanie setek mikro-obszarów zamiast kilku dużych stref i daje
    efekt „leopardzi" wzór zamiast gładkich granic.

    Args:
        quantized_f32: Mapa głębi [0, 1] float32 – używana wyłącznie do
                       odtworzenia wartości float po wygładzeniu indeksów.
        level_idx:     Mapa indeksów całkowitych 0…(n_levels-1), zwracana
                       przez quantize_depth_foreground_aware().
        n_levels:      Łączna liczba poziomów (bg_levels + fg_levels).
        kernel_size:   Rozmiar jądra morfologicznego (elipsa).  Wartości
                       7–11 są odpowiednie; większe = gładsze krawędzie,
                       ale możliwa utrata małych detali.

    Returns:
        float32 mapa głębi [0, 1] z wygładzonymi granicami poziomów.
    """
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (kernel_size, kernel_size))
    result_idx  = level_idx.copy()
    assigned    = np.zeros(level_idx.shape, dtype=bool)

    for lvl in range(n_levels):
        mask   = (level_idx == lvl).astype(np.uint8) * 255
        # CLOSE: wypełnia dziury wewnątrz obszaru danego poziomu
        mask_s = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # OPEN:  usuwa izolowane, małe wysepki (szum graniczny)
        mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN,  kernel)
        area   = (mask_s > 127) & (~assigned)
        result_idx[area] = lvl
        assigned        |= area

    # Wypełnienie ewentualnych dziur po morfologii najbliższą wartością
    if not assigned.all():
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(~assigned, return_distances=False,
                                         return_indices=True)
        # Dla każdego nieprzypisanego piksela pobierz wartość z najbliższego
        # przypisanego piksela.  indices ma kształt (2, H, W); ~assigned – maskę.
        unassigned = ~assigned
        result_idx[unassigned] = result_idx[tuple(indices[:, unassigned])]

    # Przelicz indeksy całkowite z powrotem na wartości float z quantized_f32.
    # Dla każdego indeksu pobieramy medianę wartości float z oryginalnej mapy,
    # żeby zachować dokładne wartości poziomów bez dryfu numerycznego.
    output = np.zeros_like(quantized_f32)
    for lvl in range(n_levels):
        mask_orig = (level_idx == lvl)
        mask_new  = (result_idx == lvl)
        if mask_orig.any():
            level_val = float(np.median(quantized_f32[mask_orig]))
        else:
            level_val = lvl / max(n_levels - 1, 1)
        output[mask_new] = level_val

    return output.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  LAYERED RELIEF  –  plany jako płaszczyzny zamiast surowej głębi
# ─────────────────────────────────────────────────────────────────────────────
#
# Dlaczego surowa głębia nie nadaje się do dotyku (zmierzone na „Babim lecie"
# Chełmońskiego, DPT + MiDaS, relief 7 mm):
#   • pochyły grunt od horyzontu do dolnej krawędzi zjada ~88 % wysokości –
#     palce czytają rampę, a postać jest na niej łagodnym garbem;
#   • pies siedzi na trawie w tej samej odległości co trawa obok, więc ma tę
#     samą „głębię": 15 vs 15/11 na 100, czyli ~0,3 mm – niewyczuwalne;
#   • kwantyzacja percentylowa tnie ten pochyły grunt na poziome pasy
#     (poziomice), a nie na obiekty.
#
# Dotyk potrzebuje czegoś innego: o ile obiekt wystaje ponad to, na czym stoi.
# Dlatego: (1) szacujemy powierzchnię gruntu/tła, (2) odejmujemy ją, (3) to, co
# wystaje, staje się obiektem z własną płaszczyzną i wyczuwalnym brzegiem,
# (4) grunt dostaje tylko łagodny spadek.  Bazą jest najlepsza pojedyncza mapa
# z modelu (DPT), bez syntetycznego składnika luminancyjnego, który wiąże
# wysokość z jasnością (czarny pies → dół) i nakłada fakturę trawy na cały grunt.

def _normalized_blur(values: np.ndarray, weights: np.ndarray,
                     sigma_yx: tuple) -> tuple:
    """Rozmycie gaussowskie liczone tylko z pikseli o wadze > 0 (normalized convolution)."""
    num = gaussian_filter(values * weights, sigma_yx, mode="nearest")
    den = gaussian_filter(weights, sigma_yx, mode="nearest")
    return num, den


def estimate_ground_surface(depth_f32: np.ndarray,
                            threshold:  float = 0.035,
                            exclude:    np.ndarray | None = None,
                            region:     np.ndarray | None = None,
                            work_px:    int   = 512,
                            iterations: int   = 4,
                            sigma_frac: tuple = (0.015, 0.25)) -> np.ndarray:
    """
    Szacuje powierzchnię gruntu/tła, na której „stoją" obiekty sceny.

    Grunt na obrazie zmienia głębię głównie z wysokością w kadrze (bliżej
    widza = niżej w kadrze), a powoli w poziomie.  Dlatego rozmycie jest
    anizotropowe: wąskie w pionie (σy ≈ 1,5 % wysokości – śledzi stromy
    pionowy gradient), szerokie w poziomie (σx ≈ 25 % szerokości – przeskakuje
    pod dużymi obiektami, interpolując grunt z boków).  Iteracyjnie wyklucza
    piksele, które wystają ponad bieżące oszacowanie (obiekty), więc postać
    zajmująca 85 % szerokości wiersza nie podnosi gruntu w tym wierszu.

    Start z niskiego (5.) percentyla wierszy: w wierszach zdominowanych przez
    postać wyższy percentyl trafiłby w postać i spłaszczył jej dolną część.

    Args:
        depth_f32:  Mapa głębi [0, 1], jasne = blisko.
        threshold:  Próg wystawania nad grunt (ułamek zakresu głębi).
        exclude:    Opcjonalna maska pikseli na pewno niebędących gruntem
                    (np. maski planów z GIMP-a).
        region:     Opcjonalna maska obszaru, w którym szukamy gruntu (poziom
                    zagnieżdżony: tło *wewnątrz* dużego obiektu).  Piksele spoza
                    niej nie wpływają na wynik.
        work_px:    Dłuższy bok mapy roboczej (obliczenia w zmniejszeniu).
        iterations: Liczba iteracji wykluczania obiektów.
        sigma_frac: (σy, σx) rozmycia jako ułamek wysokości/szerokości.  Domyślne
                    (wąsko w pionie, szeroko w poziomie) opisują grunt; tło we
                    wnętrzu (ściana, zasłona) zmienia się podobnie w obu
                    kierunkach, więc poziom zagnieżdżony używa izotropowego.

    Returns:
        float32 mapa gruntu w rozdzielczości wejścia.
    """
    from scipy.ndimage import binary_dilation

    H, W = depth_f32.shape
    s  = work_px / max(H, W)
    sw = max(8, int(round(W * s)))
    sh = max(8, int(round(H * s)))

    def shrink_mask(m):
        return cv2.resize(m.astype(np.uint8), (sw, sh),
                          interpolation=cv2.INTER_NEAREST) > 0

    small = cv2.resize(depth_f32.astype(np.float32), (sw, sh),
                       interpolation=cv2.INTER_AREA)
    valid = shrink_mask(region) if region is not None else np.ones((sh, sw), bool)
    if exclude is not None:
        valid &= ~shrink_mask(exclude)
    if not valid.any():
        return np.zeros((H, W), dtype=np.float32)
    sy, sx = max(1.0, sh * sigma_frac[0]), max(2.0, sw * sigma_frac[1])

    # Start: 5. percentyl każdego wiersza (tylko piksele obszaru); wiersze bez
    # pikseli obszaru dostają wartość interpolowaną z sąsiednich wierszy.
    rows = np.full(sh, np.nan, dtype=np.float32)
    for y in range(sh):
        v = small[y, valid[y]]
        if v.size:
            rows[y] = np.percentile(v, 5)
    have = ~np.isnan(rows)
    rows = np.interp(np.arange(sh), np.flatnonzero(have), rows[have]).astype(np.float32)
    ground = np.repeat(gaussian_filter(rows, sy)[:, None], sw, axis=1)

    dil = np.ones((5, 5), dtype=bool)
    for it in range(iterations):
        # Pierwsza iteracja z wyższym progiem: start z 5. percentyla zaniża
        # grunt, więc niski próg wziąłby trawę za obiekt.
        thr = threshold * (2.0 if it == 0 else 1.0)
        obj = binary_dilation((small - ground) > thr, structure=dil)
        w = (valid & ~obj).astype(np.float32)
        if not w.any():
            break
        num, den = _normalized_blur(small, w, (sy, sx))
        ground = np.where(den > 1e-3, num / np.maximum(den, 1e-6), ground)
    return cv2.resize(ground.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)


def load_plane_masks(specs, shape: tuple) -> list:
    """
    Wczytuje maski planów: lista napisów „N=ścieżka.png" (CLI) albo krotek
    (N, ścieżka).  N to numer planu w polskiej terminologii malarskiej:
    1 = pierwszy plan (najbliżej widza, najwyżej na wydruku), 2 = drugi plan…
    Biały (> 127) = należy do planu.  Kilka masek może mieć ten sam numer.

    Returns:
        lista (N, maska bool o kształcie `shape`)
    """
    out = []
    for spec in specs or []:
        if isinstance(spec, str):
            if "=" not in spec:
                raise ValueError(f"Maska planu musi mieć postać N=plik.png, dostałem: {spec}")
            num_s, path = spec.split("=", 1)
            num = int(num_s)
        else:
            num, path = spec
        buf = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"Nie można wczytać maski planu: {path}")
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = m > 127
        if mask.any():
            out.append((int(num), mask))
    return out


def pick_layered_base(depth_maps_f32: dict, fused_f32: np.ndarray) -> tuple:
    """
    Wybiera mapę bazową dla reliefu warstwowego: najlepszy pojedynczy model
    (DPT > MiDaS), a dopiero bez modeli – fuzję.  Syntetyczny składnik jest
    pomijany celowo: to odwrócona luminancja, która na obrazach daje fakturę
    pędzla i wiąże wysokość z jasnością.

    Returns:
        (mapa float32 [0, 1] w rozmiarze fuzji, nazwa źródła)
    """
    for key in ("dpt", "midas"):
        m = depth_maps_f32.get(key)
        if m is not None:
            m = m.astype(np.float32)
            if m.shape != fused_f32.shape:
                m = cv2.resize(m, (fused_f32.shape[1], fused_f32.shape[0]),
                               interpolation=cv2.INTER_CUBIC)
            return normalize_f32_robust(m), key
    return normalize_f32_robust(fused_f32), "ensemble"


def _segment_relief_objects(d: np.ndarray,
                            ground: np.ndarray,
                            region: np.ndarray | None,
                            threshold: float,
                            min_area_frac: float,
                            peak_factor: float,
                            image_area: int) -> list:
    """
    Spójne obszary wystające ponad `ground` o więcej niż `threshold`.
    Zwraca listę (maska, pole, 90. percentyl wystawania).  Odrzuca obszary
    małe oraz płaskie (90. percentyl < peak_factor·threshold) – to szerokie
    płaty gruntu, który tylko trochę odstaje od modelu, a nie obiekty.
    """
    from scipy.ndimage import (binary_opening, binary_fill_holes, label,
                               labeled_comprehension)
    from scipy.ndimage import sum as nd_sum

    H, W = d.shape
    resid = gaussian_filter(np.clip(d - ground, 0, None), max(1.0, max(H, W) / 1000))
    mask = resid > threshold
    if region is not None:
        mask &= region
    mask = binary_fill_holes(binary_opening(mask, iterations=max(1, int(round(max(H, W) / 600)))))
    lab, n = label(mask)
    out = []
    if not n:
        return out
    idx = np.arange(1, n + 1)
    areas = nd_sum(mask, lab, idx)
    peaks = labeled_comprehension(resid, lab, idx, lambda v: np.percentile(v, 90),
                                  np.float32, 0)
    for i, a, p in zip(idx, areas, peaks):
        if a >= min_area_frac * image_area and p >= peak_factor * threshold:
            out.append((lab == i, int(a), float(p)))
    return out


def _ridge_lines(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Siła wydłużonych linii (jasnych i ciemnych) w skali `sigma`: największa
    wartość własna Hessjanu minus mniejsza, czyli odpowiedź na kreskę, a nie na
    plamę.  Normalizowana sigma², więc porównywalna między skalami.
    """
    g = gaussian_filter(gray, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    tr = gxx + gyy
    disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy * gxy, 0))
    l1, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
    big = np.where(np.abs(l1) > np.abs(l2), l1, l2)
    small = np.where(np.abs(l1) > np.abs(l2), l2, l1)
    return (np.abs(big) - np.abs(small)) * sigma * sigma


def _surface_texture(gray: np.ndarray,
                     d: np.ndarray,
                     g_norm: np.ndarray,
                     obj: np.ndarray,
                     grass_amp: float = 0.025,
                     line_amp: float = 0.07,
                     cloud_amp: float = 0.04,
                     report: list | None = None) -> np.ndarray:
    """
    Faktury tła reliefu warstwowego (ułamek reliefu, do dodania do wysokości).

    Nie odtwarzają obrazu: kodują obszary dla palca, tak jak tyflografia
    rozróżnia pola fakturą.  Trzy role:
      • grunt – drobna szorstkość z pędzla; amplituda rośnie ku widzowi
        (gradient faktury to sam w sobie sygnał głębi), przy horyzoncie gaśnie;
      • linie – wydłużone kreski (patyki, źdźbła, sznurek) jako wypukłe
        grzbiety poszerzone do ~1 mm na wydruku, bo cieńszych palec nie czyta;
      • niebo – szerokie, łagodne wybrzuszenia jaśniejszych partii (chmury);
        niebo jest poza tym gładkie, więc odróżnia się od szorstkiej ziemi.
    Wszystko wygasa przy obiektach: nie dotyka ich ścian ani wierzchu.

    Amplitudy są dobrane do 7 mm reliefu i siatki 0,5 mm: 0,025 ≈ 0,2 mm na
    odchylenie standardowe (szczyty ~0,5 mm), linie ~0,5 mm, chmury ~0,3 mm.
    Mniejsze zniknęłyby w smooth_surface_for_print() (próg 0,03 reliefu).
    """
    from scipy.ndimage import distance_transform_edt, label
    from scipy.ndimage import sum as nd_sum

    H, W = gray.shape
    L = max(H, W)

    def smoothstep(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    away = smoothstep(distance_transform_edt(~obj) / max(1.0, L / 120))
    # Margines przy brzegu obrazu: skany muzealne łapią krawędź płótna albo
    # ramy.  Ciemny pasek 0,5 % szerokości przy prawym brzegu „Babiego lata"
    # robił z sąsiedniego nieba 9-milimetrowy pas wyniesiony na maksimum.
    yy = np.minimum(np.arange(H), np.arange(H)[::-1])[:, None]
    xx = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :]
    m0 = L / 200
    away = away * smoothstep((np.minimum(yy, xx) - m0) / m0)
    sky = gaussian_filter(((d < 0.01) & (g_norm < 0.01) & ~obj).astype(np.float32), L / 300)
    sky = smoothstep((sky - 0.2) / 0.6)
    ground_w = (away * (1 - sky)).astype(np.float32)
    sky_w = (away * sky).astype(np.float32)
    out = np.zeros((H, W), np.float32)

    def band(weights, s_fine, s_coarse):
        n1, d1 = _normalized_blur(gray, weights, (s_fine, s_fine))
        n2, d2 = _normalized_blur(gray, weights, (s_coarse, s_coarse))
        return n1 / np.maximum(d1, 1e-6) - n2 / np.maximum(d2, 1e-6)

    # Grunt: pasmo ~0,4–2,5 mm na wydruku 200 mm.
    if (ground_w > 0.5).sum() > 0.01 * H * W:
        s = max(1.0, L / 500)
        b = band(ground_w, s, s * 6)
        b = np.clip(b / (float(np.std(b[ground_w > 0.5])) + 1e-6), -2.5, 2.5)
        out += grass_amp * b * (0.3 + 0.7 * g_norm) * ground_w

    # Linie: grzbiety Hessjanu z progiem z histerezą – silne (górne 3 % gruntu)
    # zaczepiają, słabsze (górne 10 %) dołączają, jeśli się z nimi łączą.
    # Z samym progiem silnym kij rozpadał się na kreski.
    if (ground_w > 0.5).sum() > 0.01 * H * W:
        r = _ridge_lines(gray, max(1.0, L / 700))
        on = ground_w > 0.5
        hi_t, lo_t = np.percentile(r[on], [97, 90])
        weak = (r > lo_t) & on
        lab, n = label(weak)
        lines = np.zeros_like(weak)
        if n:
            idx = np.arange(1, n + 1)
            strong = nd_sum((r > hi_t) & weak, lab, idx)
            sizes = nd_sum(weak, lab, idx)
            keep = np.zeros(n + 1, bool)
            keep[1:] = (strong > 0) & (sizes >= max(4, L / 150))   # ≥ ~1,3 mm
            lines = keep[lab]
        width = max(1.0, L / 400)                              # ~1 mm szerokości
        ridge = gaussian_filter(lines.astype(np.float32), width)
        ridge = smoothstep(ridge / max(float(ridge.max()), 1e-6) * 3)
        out += line_amp * ridge * ground_w * (0.4 + 0.6 * g_norm)
        if report is not None:
            report.append(f"faktura: linie {100 * lines.mean():.1f} % gruntu")

    # Niebo: duże, miękkie formy, jaśniejsze = wyżej.
    if (sky_w > 0.5).sum() > 0.01 * H * W:
        s = max(1.0, L / 120)
        b = band(sky_w, s, s * 5)
        b = np.clip(b / (float(np.std(b[sky_w > 0.5])) + 1e-6), -2.0, 2.0)
        b = np.maximum(b, -0.5)
        b -= float(b[sky_w > 0.5].mean())     # inaczej całe niebo się podnosi: uskok na granicy
        out += cloud_amp * b * sky_w

    return out


def build_layered_relief(depth_f32: np.ndarray,
                         image_bgr:       np.ndarray | None = None,
                         plane_masks:     list | None = None,
                         threshold:       float = 0.035,
                         min_area_frac:   float = 0.002,
                         peak_factor:     float = 2.5,
                         nest_frac:       float = 0.35,
                         ground_span:     float = 0.20,
                         plateau_base:    float = 0.30,
                         level_step:      float = 0.25,
                         nearness_span:   float = 0.15,
                         interior_gain:   float = 0.30,
                         dome_gain:       float = 0.25,
                         dome_radius:     float = 0.04,
                         edge_soft:       float = 0.025,
                         detail_strength: float = 0.08,
                         detail_scale:    float = 1.0 / 400,
                         texture:         float = 1.0,
                         work_px:         int   = 2048,
                         report:          list | None = None) -> np.ndarray:
    """
    Buduje relief warstwowy do druku dotykowego: grunt jako łagodny spadek,
    każdy obiekt jako osobna płaszczyzna z wyczuwalnym brzegiem i własnym,
    ściśniętym kształtem wewnętrznym.  Brzeg to łagodne zbocze szerokości
    edge_soft·dłuższy bok (0,025 ≈ 5 mm przy 200 mm), wyśrodkowane na konturze;
    edge_soft = 0 daje pionową ścianę.

    Wysokości (ułamek reliefu; na końcu skalowane tak, by maksimum = 1):
      • niebo / najdalszy grunt   0
      • grunt                     0 … ground_span       (łagodny spadek ku widzowi)
      • obiekt poziomu L          plateau_base + level_step·(L−1)
                                  + nearness_span·bliskość gruntu pod obiektem
      • wnętrze obiektu           + interior_gain · kształt z mapy głębi
      • zaokrąglenie              + dome_gain · parabola t(2−t) od konturu
                                  (promień ≤ dome_radius dłuższego boku)
      • detal (tylko w obiektach) ± detail_strength     (fałdy, kontury kończyn)

    Obiekty: z masek planów (gdy podane – wskazane przez człowieka, plan 1 =
    pierwszy plan = najwyżej) albo z automatycznej segmentacji.  Segmentacja
    jest zagnieżdżona: obszar większy niż `nest_frac` obrazu to raczej ściana,
    zasłona albo całe wnętrze niż postać (we wnętrzach najdalszy w wierszu
    jest zwykle otwór okna, więc wszystko inne „wystaje").  Wtedy grunt jest
    szacowany ponownie *wewnątrz* tego obszaru, a to, co wystaje dalej
    (postać na tle zasłony), dostaje poziom wyżej.  Głębokość zagnieżdżenia: 2.

    Detal z luminancji jest pasmowo-przepustowy i liczony WYŁĄCZNIE z pikseli
    obiektów (normalized convolution): inaczej jasna trawa wokół czarnego psa
    przesuwa średnią i wciska psa w dół.  Skala pasma to ułamek dłuższego boku
    obrazu, więc jest fizyczna na wydruku (1/400 ≈ 0,5 mm przy 200 mm).

    Obliczenia biegną w rozdzielczości co najwyżej `work_px` (dłuższy bok);
    wynik jest skalowany z powrotem.  Siatka STL ma i tak ≤ 512 px.

    Args:
        depth_f32:     Mapa bazowa [0, 1], jasne = blisko (pick_layered_base).
        image_bgr:     Oryginał BGR do detalu; None = bez detalu.
        plane_masks:   Wynik load_plane_masks(); None/[] = segmentacja automatyczna.
        report:        Opcjonalna lista, do której trafiają opisy obiektów.

    Returns:
        float32 [0, 1] – mapa wysokości gotowa do depth_to_stl() (bez kwantyzacji
        do uint8: na łagodnym gruncie 256 poziomów drukuje się jako poziomice).
    """
    from scipy.ndimage import distance_transform_edt

    full_h, full_w = depth_f32.shape
    scale = min(1.0, work_px / max(full_h, full_w))
    H, W = max(8, int(round(full_h * scale))), max(8, int(round(full_w * scale)))

    def to_work(a, interp):
        return a if a.shape == (H, W) else cv2.resize(a, (W, H), interpolation=interp)

    d = normalize_f32_robust(to_work(depth_f32.astype(np.float32), cv2.INTER_AREA))
    image_area = H * W

    masks = [(n, to_work(m.astype(np.uint8), cv2.INTER_NEAREST) > 0)
             for n, m in (plane_masks or [])]
    exclude = None
    if masks:
        exclude = np.zeros((H, W), dtype=bool)
        for _, m in masks:
            exclude |= m
    ground = estimate_ground_surface(d, threshold=threshold, exclude=exclude)
    g_lo, g_hi = np.percentile(ground, [1, 99])
    g_norm = np.clip((ground - g_lo) / max(g_hi - g_lo, 1e-6), 0, 1)

    height = (ground_span * g_norm).astype(np.float32)
    objects = []    # (maska, wysokość płaszczyzny, opis)

    if masks:
        nums = sorted({n for n, _ in masks})
        for n, m in masks:
            rank = nums.index(n)                         # 0 = plan 1 = najbliżej
            level = len(nums) - rank                     # plan 1 → najwyższy poziom
            objects.append((m, plateau_base + level_step * (level - 1),
                            f"plan {n}: {int(m.sum())} px"))
    else:
        def descend(region, region_ground, level):
            for m, area, peak in _segment_relief_objects(
                    d, region_ground, region, threshold, min_area_frac,
                    peak_factor, image_area):
                near = float(g_norm[m].mean())
                plateau = plateau_base + level_step * (level - 1) + nearness_span * near
                objects.append((m, plateau,
                                f"poziom {level}: {100 * area / image_area:.1f} % obrazu, "
                                f"wystaje {peak:.2f}, bliskość {near:.2f}"))
                if level < 2 and area > nest_frac * image_area:
                    inner_ground = estimate_ground_surface(d, threshold=threshold, region=m,
                                                          sigma_frac=(0.06, 0.06))
                    descend(m, inner_ground, level + 1)
        descend(None, ground, 1)

    # Kształt wewnętrzny: głębia w obrębie obiektu, wspólna skala dla wszystkich
    # obiektów – mały, płaski pies nie dostaje rozciągniętego szumu.  Obiekty
    # zagnieżdżone są na liście po rodzicu, więc nadpisują go (np.maximum).
    ranges = [float(np.percentile(d[m], 98) - np.percentile(d[m], 2))
              for m, _, _ in objects if m.any()]
    span = max(ranges) if ranges else 1.0
    for m, plateau, desc in objects:
        if not m.any():
            continue
        base = float(np.percentile(d[m], 2))
        inner = np.clip((d - base) / max(span, 1e-6), 0, 1)
        # Zaokrąglenie: parabola t(2−t) od konturu do promienia R.  Obiekt ma
        # być wypukły pod palcem – płaska płyta ze ścianą czyta się jak taca,
        # a mapa głębi czarnej sylwetki bywa płaska.  R ≤ 4 % dłuższego boku
        # (≈ 8 mm przy 200 mm), więc wielka zasłona dostaje tylko zaokrąglony
        # brzeg, a nie poduszkę na pół obrazu.
        dt = distance_transform_edt(m)
        R = max(1.0, min(float(np.percentile(dt[m], 95)), max(H, W) * dome_radius))
        t = np.clip(dt / R, 0, 1)
        dome = t * (2.0 - t)
        top = plateau + interior_gain * inner + dome_gain * dome
        if edge_soft > 0:
            # Łagodne podejście od gruntu zamiast pionowej ściany: zbocze
            # szerokości w, wyśrodkowane na konturze (pół na trawie, pół w
            # obiekcie), styczne do gruntu u dołu i zaokrąglone u góry.  Samo
            # wewnętrzne zbocze 3 mm przy 4 mm wysokości miało nadal ~55° i
            # zjadało wąskie części (rękę); wyśrodkowane traci połowę mniej.
            w = max(1.0, max(H, W) * edge_soft)
            dt_out, (iy, ix) = distance_transform_edt(~m, return_indices=True)
            top_ext = np.where(m, top, top[iy, ix])          # wierzch przedłużony poza kontur
            e = np.clip((dt - dt_out + w / 2) / w, 0, 1)
            e = e * e * (3 - 2 * e)
            ramp = height + (top_ext - height) * e
            zone = m | (dt_out < w / 2)
            height = np.where(zone, np.maximum(height, ramp), height)
        else:
            height = np.where(m, np.maximum(height, top), height)
        if report is not None:
            report.append(f"{desc}, płaszczyzna {plateau:.2f}")

    if image_bgr is not None and detail_strength > 0 and objects:
        obj = np.zeros((H, W), dtype=bool)
        for m, _, _ in objects:
            obj |= m
        # Detal wygasa ku konturowi i nie bierze z niego próbek: maska nigdy
        # nie leży idealnie na sylwetce, a piksele z jej brzegu to zwykle tło
        # (jasne niebo za czarnym psem) albo futro pod światło.  Liczone wprost
        # podnosiły obwódkę o ~0,6 mm ponad środek – wierzch wyglądał jak taca.
        edge = max(1.0, max(H, W) / 120)
        feather = np.clip(distance_transform_edt(obj) / edge, 0, 1).astype(np.float32)
        feather = feather * feather * (3 - 2 * feather)          # smoothstep
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = to_work(gray, cv2.INTER_AREA)
        sg = max(1.0, max(H, W) * detail_scale)
        n1, d1 = _normalized_blur(gray, feather, (sg, sg))
        n2, d2 = _normalized_blur(gray, feather, (sg * 8, sg * 8))
        band = n1 / np.maximum(d1, 1e-6) - n2 / np.maximum(d2, 1e-6)
        inside = feather > 0.5
        band = band / (float(np.std(band[inside])) + 1e-6) if inside.any() else band * 0
        height = height + np.clip(band * detail_strength, -0.15, 0.15) * feather

    height = median_filter(height, size=5)
    height = gaussian_filter(height, max(1.0, max(H, W) / 700))
    if image_bgr is not None and texture > 0:
        # Po wygładzeniu, nie przed: mediana 5 px zjadłaby źdźbła i patyki.
        obj = np.zeros((H, W), dtype=bool)
        for m, _, _ in objects:
            obj |= m
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        height = height + texture * _surface_texture(
            to_work(gray, cv2.INTER_AREA), d, g_norm, obj, report=report)
    height = np.clip(height, 0, None)
    height = height / max(float(height.max()), 1e-6)
    if (H, W) != (full_h, full_w):
        height = cv2.resize(height, (full_w, full_h), interpolation=cv2.INTER_CUBIC)
    return np.clip(height, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  STL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _height_to_unit_float(depth: np.ndarray) -> np.ndarray:
    """uint8 / uint16 / float → float32 [0, 1] bez utraty precyzji."""
    if depth.dtype == np.uint8:
        return depth.astype(np.float32) / 255.0
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 65535.0
    return np.clip(depth.astype(np.float32), 0.0, 1.0)


def smooth_surface_for_print(height_f32: np.ndarray,
                             strength: float = 0.03,
                             radius_px: int = 2) -> np.ndarray:
    """
    Wygładzanie zachowujące krawędzie, na siatce STL (filtr bilateralny).

    Wygładza drobne falowanie, którego amplituda jest mniejsza od `strength`
    (ułamek reliefu; 0,03 przy 7 mm ≈ 0,2 mm): szum mapy, pozostałości faktury.
    Uskoki większe od niego – ściany planów, kontury postaci – zostają strome,
    bo filtr nie uśrednia pikseli o bardzo różnej wysokości.

    Args:
        height_f32: Mapa wysokości [0, 1] w rozdzielczości siatki.
        strength:   Próg „to jeszcze szum" w jednostkach [0, 1]; 0 = wyłączone.
        radius_px:  Promień przestrzenny w komórkach siatki.
    """
    if strength <= 0:
        return height_f32
    out = height_f32.astype(np.float32)
    for _ in range(2):   # dwa łagodne przejścia gładzą lepiej niż jedno silne
        out = cv2.bilateralFilter(out, d=2 * radius_px + 1,
                                  sigmaColor=float(strength),
                                  sigmaSpace=float(radius_px))
    return out


def depth_to_stl(depth_u8: np.ndarray,
                 output_path: str,
                 width_mm:     float = 200.0,
                 relief_mm:    float = 10.0,
                 base_mm:      float = 3.0,
                 max_px:       int   = 512,
                 invert_depth: bool  = False,
                 flip_x:       bool  = False,
                 flip_y:       bool  = True,
                 smooth:       float = 0.03) -> None:
    """
    Converts a height map to a binary, watertight STL for 3D printing.

    The model consists of:
      • Top surface (relief) – triangle mesh with height from the map
      • Four side walls and a bottom closing the solid

    Surface quality (what used to read as "dense, jagged triangles"):
      • float input is used as-is – a uint8 map has only 256 heights, which on
        a gentle slope prints as visible contour lines (0.03 mm terraces);
      • downscaling uses area averaging, not Lanczos, whose overshoot left a
        ripple along every steep wall;
      • each grid quad is split along the diagonal that follows the surface
        (the pair of corners with the closer heights), so diagonal contours
        do not zigzag;
      • smooth_surface_for_print() removes sub-threshold undulation while
        keeping walls steep.
    All faces are oriented outwards and the bottom is a fan sharing every
    perimeter vertex of the walls (no T-junctions), so the mesh is closed.

    Physical dimensions:
      • Width:  width_mm  [mm]
      • Depth:  auto (preserves image aspect ratio)
      • Height: base_mm + relief_mm [mm]

    Args:
        depth_u8:     Height map: uint8 [0-255], uint16 [0-65535] or float [0, 1]
                      (float preferred – no quantisation)
        output_path:  Output path (.stl)
        width_mm:     Model width [mm]
        relief_mm:    Maximum relief height above base [mm]
        base_mm:      Base plate thickness [mm]
        max_px:       Maximum mesh resolution (longer side) [px]
        invert_depth: Invert heights
        flip_x:       Horizontal mirror
        flip_y:       Vertical mirror (default True –
                      corrects image→Prusa Slicer Y orientation)
        smooth:       Edge-preserving smoothing threshold (fraction of relief;
                      0 = off).  See smooth_surface_for_print().
    """
    d = _height_to_unit_float(depth_u8)
    if invert_depth:
        d = 1.0 - d
    if flip_x:
        d = np.fliplr(d)
    if flip_y:
        d = np.flipud(d)
    d = np.ascontiguousarray(d)

    # ── Mesh scaling ──────────────────────────────────────────────────────────
    orig_h, orig_w = d.shape
    if max(orig_h, orig_w) > max_px:
        scale = max_px / max(orig_h, orig_w)
        new_w = max(2, int(orig_w * scale))
        new_h = max(2, int(orig_h * scale))
        d = cv2.resize(d, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rows, cols = d.shape
    d = np.clip(smooth_surface_for_print(d, strength=smooth), 0.0, 1.0)

    height_mm  = width_mm * rows / cols
    px_to_mm_x = width_mm  / (cols - 1)
    px_to_mm_y = height_mm / (rows - 1)

    xs = np.arange(cols, dtype=np.float32) * px_to_mm_x
    ys = np.arange(rows, dtype=np.float32) * px_to_mm_y
    Z  = (base_mm + d * relief_mm).astype(np.float32)

    logger.info(f"STL mesh: {cols}×{rows} px  →  "
                f"{width_mm:.1f}×{height_mm:.1f}×{base_mm + relief_mm:.1f} mm")

    # ── Top surface ───────────────────────────────────────────────────────────
    X, Y = np.meshgrid(xs, ys)
    top = np.stack([X, Y, Z], axis=-1)                     # (rows, cols, 3)
    a = top[:-1, :-1].reshape(-1, 3)                       # (c,   r)
    b = top[:-1, 1:].reshape(-1, 3)                        # (c+1, r)
    c = top[1:, :-1].reshape(-1, 3)                        # (c,   r+1)
    e = top[1:, 1:].reshape(-1, 3)                         # (c+1, r+1)
    # Split along the diagonal whose end heights are closer – it follows the
    # ridge/valley instead of cutting across it.
    use_ad = (np.abs(a[:, 2] - e[:, 2]) < np.abs(b[:, 2] - c[:, 2]))[:, None, None]
    t1 = np.where(use_ad, np.stack([a, b, e], 1), np.stack([a, b, c], 1))
    t2 = np.where(use_ad, np.stack([a, e, c], 1), np.stack([b, e, c], 1))
    parts = [t1, t2]                                        # both +z

    # ── Walls ─────────────────────────────────────────────────────────────────
    def wall(tops: np.ndarray, outward: np.ndarray) -> np.ndarray:
        bots = tops.copy()
        bots[:, 2] = 0.0
        p, q = tops[:-1], tops[1:]
        pb, qb = bots[:-1], bots[1:]
        tri = np.concatenate([np.stack([p, q, pb], 1), np.stack([pb, q, qb], 1)])
        return _orient(tri, outward)

    parts.append(wall(top[:, 0],  np.array([-1, 0, 0], np.float32)))   # left
    parts.append(wall(top[:, -1], np.array([ 1, 0, 0], np.float32)))   # right
    parts.append(wall(top[0, :],  np.array([0, -1, 0], np.float32)))   # front
    parts.append(wall(top[-1, :], np.array([0,  1, 0], np.float32)))   # back

    # ── Bottom: fan from the centre over every perimeter edge ─────────────────
    perim = np.concatenate([top[0, :], top[1:, -1], top[-1, -2::-1], top[-2:0:-1, 0]])
    perim = perim.copy()
    perim[:, 2] = 0.0
    nxt = np.roll(perim, -1, axis=0)
    centre = np.broadcast_to(np.array([xs[-1] / 2, ys[-1] / 2, 0.0], np.float32),
                             perim.shape)
    bottom = np.stack([centre, perim, nxt], 1)
    parts.append(_orient(bottom, np.array([0, 0, -1], np.float32)))

    tris = np.concatenate(parts).astype(np.float32)
    all_triangles = np.zeros(len(tris), dtype=stl_mesh.Mesh.dtype)
    all_triangles['vectors'] = tris
    m = stl_mesh.Mesh(all_triangles)
    m.update_normals()
    m.save(output_path, mode=StlMode.BINARY)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"STL saved: {output_path}  ({size_mb:.1f} MB, {len(tris)} triangles)")


def _orient(tris: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Flips triangles whose winding normal points against `outward`."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (n @ outward) < 0
    out = tris.copy()
    out[flip, 1], out[flip, 2] = tris[flip, 2], tris[flip, 1]
    # Degenerate wall triangles (zero height at an edge) are harmless; keep them.
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_path: str,
                 output_dir: str     = "output",
                 width_mm:   float   = 200.0,
                 relief_mm:  float   = 10.0,
                 base_mm:    float   = 3.0,
                 mesh_px:    int     = 512,
                 clahe_clip: float   = 2.0,
                 smooth_sigma: float = 0.7,
                 invert_depth: bool  = False,
                 flip_x:       bool  = False,
                 flip_y:       bool  = True,
                 detail_strength:    float = 0.15,
                 detail_blur_sigma:  float = 1.2,
                 tactile:            bool  = False,
                 tactile_median_size: int  = 5,
                 tactile_sigma:      float = 3.5,
                 tactile_levels:     int   = 0,
                 tactile_fg_threshold_pct: float = 40.0,
                 tactile_bg_levels:        int   = 2,
                 tactile_fg_levels:        int   = 4,
                 tactile_boundary_kernel:  int   = 9,
                 tactile_multiscale:       bool  = False,
                 tactile_fine_sigma:       float = 1.5,
                 tactile_limb_sigma:       float = 3.0,
                 fill_holes:               bool  = False,
                 fill_holes_min_area:      int   = 20,
                 fill_holes_max_area:      int   = 2000,
                 fill_holes_kernel:        int   = 5,
                 tactile_layered:          bool  = False,
                 plane_masks:              list | None = None,
                 layered_threshold:        float = 0.035,
                 layered_detail:           float = 0.08,
                 layered_texture:          float = 1.0,
                 layered_edge_mm:          float = 5.0,
                 stl_smooth:               float = 0.03) -> None:
    """
    Runs the full pipeline:
      image → 3× depth map → fusion → (detail overlay OR tactile smoothing) → STL

    Args:
        input_path:                Path to the input image
        output_dir:                Output directory
        width_mm:                  STL model width [mm]
        relief_mm:                 Maximum relief height [mm]
        base_mm:                   Base plate thickness [mm]
        mesh_px:                   Maximum STL mesh resolution [px]
        clahe_clip:                CLAHE clip limit (default 2.0; set 0 for tactile)
        smooth_sigma:              Gaussian σ [px] for final noise pass
        invert_depth:              Invert depth values
        flip_x:                    Horizontal mirror
        flip_y:                    Vertical mirror (default True for Prusa)
        detail_strength:           Micro-detail overlay amplitude (0 = disabled).
                                   In tactile mode, overlay runs BEFORE smoothing —
                                   use low values (0.05–0.08) with high blur_sigma
                                   (2.0–2.5) to recover limb contours without spikes.
        detail_blur_sigma:         Low-pass cutoff for detail extraction [px].
                                   Higher values (2.0–2.5) extract broader shadow bands
                                   (limb silhouettes) while suppressing fine spikes
                                   (thread, eyelashes).  Default 1.2 for visual mode.
        tactile:                   Enable tyflographic / tactile mode.  Overrides
                                   detail_strength=0, applies prepare_for_touch()
                                   and disables CLAHE.  Optimised for touch not sight.
        tactile_median_size:       Median filter footprint for spike removal [px]
                                   (tactile mode only, default 5)
        tactile_sigma:             Gaussian σ for broad tactile smoothing [px]
                                   (tactile mode only, default 3.5)
        tactile_levels:            If > 1, quantize the depth map using
                                   quantize_depth_foreground_aware() with
                                   (tactile_bg_levels + tactile_fg_levels) total
                                   levels, then smooth boundaries morphologically.
                                   0 = continuous gradient (default).
        tactile_fg_threshold_pct:  Percentile separating background from foreground
                                   in foreground-aware quantization (default 40.0).
        tactile_bg_levels:         Discrete levels for the background zone (default 2).
        tactile_fg_levels:         Discrete levels for the foreground zone (default 4).
        tactile_boundary_kernel:   Morphological kernel size for boundary smoothing
                                   (default 9 px).
        tactile_multiscale:        Use multi-scale smoothing (prepare_for_touch_multiscale)
                                   instead of single-pass Gaussian.  Preserves limb-scale
                                   contours while removing fine texture noise (default False).
        tactile_fine_sigma:        σ [px] for fine-texture removal in multiscale mode
                                   (default 1.5; removes fabric/grass noise).
        tactile_limb_sigma:        σ [px] defining limb-scale in multiscale mode
                                   (default 3.0; final filter uses limb_sigma*0.5 to
                                   avoid merging adjacent legs/arms).
        fill_holes:                Run fill_small_object_holes() after fusion to patch
                                   small objects (animals, distant figures) whose interiors
                                   are flat/empty in the ensemble depth map (default False).
        fill_holes_min_area:       Minimum contour area [px²] to consider (default 20).
        fill_holes_max_area:       Maximum contour area [px²] to consider (default 2000).
        fill_holes_kernel:         Morphological kernel size for contour closing (default 5).
        tactile_layered:           Layered relief for touch (build_layered_relief):
                                   ground removed, every object a raised plane with
                                   a palpable wall.  Implies tactile; replaces
                                   fill_holes / detail overlay / multiscale /
                                   quantization, which act on raw depth.
        plane_masks:               ["N=mask.png", ...] – plane masks for layered
                                   relief (1 = foreground, highest).  None = automatic
                                   object segmentation.
        layered_threshold:         How far above the ground a region must rise to be
                                   an object (fraction of depth range, default 0.035).
        layered_detail:            Amplitude of luminance detail inside objects
                                   (folds, limb contours; 0 = off, default 0.08).
        layered_edge_mm:           Width of the slope where an object meets the
                                   ground, centred on the contour, in print mm
                                   (0 = vertical wall, default 5).
        layered_texture:           Background textures – grass roughness, linear
                                   features (sticks, stalks), soft clouds; a
                                   multiplier (0 = smooth ground and sky, default 1).
        stl_smooth:                Edge-preserving smoothing on the STL grid
                                   (fraction of relief; 0 = off, default 0.03).
    """
    if tactile_layered:
        tactile = True
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
    # Use np.frombuffer + imdecode to handle Unicode/non-ASCII paths on Windows
    _buf = np.frombuffer(open(str(img_path), "rb").read(), dtype=np.uint8)
    image = cv2.imdecode(_buf, cv2.IMREAD_COLOR)
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
                           df._run_openvino_inference_f32(_c, img, "midas")))
    if dpt_compiled:
        tasks.append(("dpt",
                       lambda img, _c=dpt_compiled:
                           df._run_openvino_inference_f32(_c, img, "dpt")))

    # OpenVINO is not thread-safe – sequential execution
    labels_map = {"standard": "Standard (synthetic)",
                  "midas":    "OpenVINO MiDaS v2.1 Small",
                  "dpt":      "OpenVINO DPT Large"}

    depth_maps_f32 = {}   # float32 – used for scale-shift fusion (raw model output)
    depth_maps_u8  = {}   # uint8  – used for display / individual PNG saving

    for name, fn in tasks:
        name, depth_raw, ms = run_method(name, fn)
        times[name] = ms

        # Standard model already returns uint8; OpenVINO models return float32
        if depth_raw.dtype != np.float32:
            depth_f32 = depth_raw.astype(np.float32)
        else:
            depth_f32 = depth_raw

        depth_maps_f32[name] = depth_f32

        # Normalise to uint8 [0-255] for display and individual file saving
        # (percentile-robust stretch avoids outlier-induced flattening)
        depth_u8 = (normalize_f32_robust(depth_f32) * 255).astype(np.uint8)
        depth_maps_u8[name] = depth_u8

        print(f"  ✓ {labels_map.get(name, name):<30}  {ms:7.1f} ms")

        # Save individual maps – depth as lossless PNG, colour as PNG
        cv2.imwrite(str(out_dir / f"{stem}_depth_{name}.png"), depth_u8)
        cv2.imwrite(
            str(out_dir / f"{stem}_depth_{name}_color.png"),
            cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO),
        )

    # ── 4. Fusion (scale-shift alignment) ────────────────────────────────────
    print(f"\n  {'─'*64}")
    print(f"  Fusing depth maps (scale-shift ensemble)…")

    active_weights = {k: ENSEMBLE_WEIGHTS[k] for k in depth_maps_f32}
    total_w = sum(active_weights.values())
    for k, w_val in active_weights.items():
        pct = w_val / total_w * 100
        print(f"    {k:<10}  weight {w_val:.2f}  ({pct:.0f} %)")

    fused_f32 = fuse_depth_maps(depth_maps_f32)

    # ── 4a. (opcjonalne) Wypełnienie wnętrz małych obiektów ───────────────────
    # Modele DPT/MiDaS przy małych obiektach (pies, daleka postać) wykrywają
    # tylko kontur, a wnętrze jest płaskie/puste.  fill_small_object_holes()
    # podnosi wnętrze do poziomu 75. percentyla pikseli brzegowych.
    stl_source = None   # float height map for STL; None → use the uint8 ensemble
    if fill_holes and not tactile_layered:
        t_fh = time.perf_counter()
        fused_f32 = fill_small_object_holes(
            fused_f32,
            min_area    = fill_holes_min_area,
            max_area    = fill_holes_max_area,
            kernel_size = fill_holes_kernel,
        )
        ms_fh = (time.perf_counter() - t_fh) * 1000
        print(f"    fill_small_object_holes: area {fill_holes_min_area}–"
              f"{fill_holes_max_area} px², kernel={fill_holes_kernel}  "
              f"({ms_fh:.0f} ms)")
    if tactile_layered:
        # ── LAYERED RELIEF PATH ───────────────────────────────────────────────
        print(f"\n  {'─'*64}")
        print(f"  Layered relief (planes for touch)…")
        t_lay = time.perf_counter()
        base_f32, base_name = pick_layered_base(depth_maps_f32, fused_f32)
        masks = load_plane_masks(plane_masks, base_f32.shape) if plane_masks else []
        report: list = []
        relief_f32 = build_layered_relief(base_f32, image,
                                          plane_masks     = masks,
                                          threshold       = layered_threshold,
                                          detail_strength = layered_detail,
                                          texture         = layered_texture,
                                          edge_soft       = max(0.0, layered_edge_mm) / max(width_mm, 1e-6),
                                          report          = report)
        print(f"    base map: {base_name}   "
              + (f"plane masks: {len(masks)}" if masks else "automatic segmentation"))
        for line in report:
            print(f"    {line}")
        if not report:
            print("    no object rises above the ground – only the ground slope is printed")
        relief_u16 = (relief_f32 * 65535).astype(np.uint16)
        cv2.imwrite(str(out_dir / f"{stem}_relief_layered.png"), relief_u16)
        print(f"  ✓ Layered relief built ({(time.perf_counter() - t_lay) * 1000:.0f} ms)")
        fused_f32  = relief_f32
        stl_source = relief_f32
        clahe_clip = 0.0
        smooth_sigma = 0.0
    elif tactile:
        # ── TACTILE PATH ──────────────────────────────────────────────────────
        # Kolejność kroków:
        #   1. (opcjonalnie) detail_overlay – wyciąga kontury kończyn z luminancji
        #      obrazu zanim ensemble je "zgubił"; nawet niski detail_strength (0.05–0.08)
        #      z wysokim blur_sigma (2.0–2.5) przywraca krawędź nogi bez igieł.
        #   2. prepare_for_touch / multiscale – usuwa drobny szum tekstury;
        #      działa NA mapie z nakładką, więc igły z overlay też są wygładzane.
        #   3. Opcjonalna kwantyzacja + wygładzanie granic morfologiczne.
        #
        # Uwaga: detail_strength=0 (domyślne w run_pipeline_tactile) całkowicie
        # pomija krok 1, zachowując dotychczasowe zachowanie.
        print(f"\n  {'─'*64}")
        print(f"  Tactile / tyflographic processing…")
        t_tact = time.perf_counter()

        # Krok 1: detail overlay przed wygładzaniem taktylnym (opcjonalny)
        if detail_strength > 0.0:
            fused_f32 = apply_detail_overlay(fused_f32, image,
                                             strength=detail_strength,
                                             blur_sigma=detail_blur_sigma)
            det_u8 = (normalize_f32_robust(fused_f32) * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{stem}_depth_ensemble_detail.png"), det_u8)
            cv2.imwrite(
                str(out_dir / f"{stem}_depth_ensemble_detail_color.png"),
                cv2.applyColorMap(det_u8, cv2.COLORMAP_INFERNO),
            )
            print(f"    Detail overlay applied before smoothing "
                  f"(strength={detail_strength}, blur_sigma={detail_blur_sigma})")

        # Krok 2: wygładzanie taktylne (usuwa szum tkaniny + ewentualne igły overlay)
        if tactile_multiscale:
            fused_f32 = prepare_for_touch_multiscale(
                fused_f32,
                median_size = tactile_median_size,
                fine_sigma  = tactile_fine_sigma,
                limb_sigma  = tactile_limb_sigma,
            )
            smooth_label = (f"multiscale (fine_σ={tactile_fine_sigma}, "
                            f"limb_σ={tactile_limb_sigma})")
        else:
            fused_f32 = prepare_for_touch(fused_f32,
                                          median_size=tactile_median_size,
                                          sigma=tactile_sigma)
            smooth_label = f"single-pass σ={tactile_sigma}"
        if tactile_levels > 1:
            # Krok 2: Foreground-aware quantization – postać dostaje więcej poziomów.
            # Funkcja zwraca (float_map, level_idx, n_levels) – indeksy całkowite
            # są wymagane przez smooth_quantized_boundaries() do poprawnej pracy.
            fused_f32, _level_idx, _n_levels = quantize_depth_foreground_aware(
                fused_f32,
                fg_threshold_pct = tactile_fg_threshold_pct,
                bg_levels        = tactile_bg_levels,
                fg_levels        = tactile_fg_levels,
            )
            total_levels = tactile_bg_levels + tactile_fg_levels
            print(f"    Depth quantized: {tactile_bg_levels} bg + {tactile_fg_levels} fg "
                  f"= {total_levels} levels  (fg threshold: {tactile_fg_threshold_pct:.0f} pct)")

            # Krok 3: Wygładzenie postrzępionych krawędzi między poziomami.
            # Pomijamy gdy kernel_size <= 0.
            if tactile_boundary_kernel > 0:
                fused_f32 = smooth_quantized_boundaries(
                    fused_f32,
                    level_idx   = _level_idx,
                    n_levels    = _n_levels,
                    kernel_size = tactile_boundary_kernel,
                )
                print(f"    Boundary smoothing applied "
                      f"(morphological, kernel={tactile_boundary_kernel})")
            else:
                print(f"    Boundary smoothing skipped (kernel=0)")
        ms_tact = (time.perf_counter() - t_tact) * 1000
        # Save tactile intermediate PNG for verification
        tact_u8 = (normalize_f32_robust(fused_f32) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_depth_ensemble_tactile.png"), tact_u8)
        cv2.imwrite(
            str(out_dir / f"{stem}_depth_ensemble_tactile_color.png"),
            cv2.applyColorMap(tact_u8, cv2.COLORMAP_INFERNO),
        )
        print(f"  ✓ Tactile smoothing applied ({smooth_label}"
              + (f", bg_levels={tactile_bg_levels}, fg_levels={tactile_fg_levels}"
                 if tactile_levels > 1 else "")
              + f", {ms_tact:.0f} ms)")
        # STL from float, not from the uint8 ensemble: 256 levels print as
        # contour lines on gentle slopes.  Same tone curve as postprocess_depth.
        stl_source = normalize_f32_robust(fused_f32) ** 1.02
        # Force CLAHE off – local contrast enhancement has no meaning in touch
        clahe_clip = 0.0
        # Sigma already handled by prepare_for_touch; keep post-process pass minimal
        smooth_sigma = 0.0
    else:
        # ── VISUAL PATH – micro-detail overlay ───────────────────────────────
        # The ensemble captures coarse geometry; this step re-injects high-
        # frequency surface texture (fabric, skin, ground) extracted directly
        # from image luminance, so the two concerns are handled independently.
        # In tactile mode, detail overlay is handled in the tactile block above.
        if detail_strength > 0.0:
            t_det = time.perf_counter()
            fused_f32 = apply_detail_overlay(fused_f32, image,
                                             strength=detail_strength,
                                             blur_sigma=detail_blur_sigma)
            ms_det = (time.perf_counter() - t_det) * 1000
            # Save intermediate for visual inspection
            det_u8 = (normalize_f32_robust(fused_f32) * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{stem}_depth_ensemble_detail.png"), det_u8)
            cv2.imwrite(
                str(out_dir / f"{stem}_depth_ensemble_detail_color.png"),
                cv2.applyColorMap(det_u8, cv2.COLORMAP_INFERNO),
            )
            print(f"  ✓ Micro-detail overlay applied "
                  f"(strength={detail_strength}, blur_sigma={detail_blur_sigma}, "
                  f"{ms_det:.0f} ms)")
        else:
            print(f"  ─ Micro-detail overlay disabled (strength=0)")

    # Light Gaussian on top of self-guided filter for final noise removal
    fused_post = postprocess_depth(fused_f32, clahe_clip=clahe_clip, sigma=smooth_sigma)

    # Save ensemble – lossless PNG for both depth and colour map
    cv2.imwrite(str(out_dir / f"{stem}_depth_ensemble.png"), fused_post)
    cv2.imwrite(
        str(out_dir / f"{stem}_depth_ensemble_color.png"),
        cv2.applyColorMap(fused_post, cv2.COLORMAP_INFERNO),
    )
    print(f"  ✓ Ensemble saved")

    # ── 5. Comparison grid ────────────────────────────────────────────────────
    _save_comparison(image, depth_maps_u8, fused_post, out_dir, stem)
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
    depth_to_stl(stl_source if stl_source is not None else fused_post,
                 stl_path,
                 width_mm     = width_mm,
                 relief_mm    = relief_mm,
                 base_mm      = base_mm,
                 max_px       = mesh_px,
                 invert_depth = invert_depth,
                 flip_x       = flip_x,
                 flip_y       = flip_y,
                 smooth       = stl_smooth)
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


def run_pipeline_tactile(input_path: str,
                         output_dir: str   = "output",
                         width_mm:   float = 200.0,
                         relief_mm:  float = 7.0,
                         base_mm:    float = 3.0,
                         mesh_px:    int   = 140,
                         tactile_median_size:      int   = 5,
                         tactile_sigma:            float = 3.5,
                         tactile_levels:           int   = 0,
                         tactile_fg_threshold_pct: float = 40.0,
                         tactile_bg_levels:        int   = 2,
                         tactile_fg_levels:        int   = 4,
                         tactile_boundary_kernel:  int   = 9,
                         tactile_multiscale:       bool  = False,
                         tactile_fine_sigma:       float = 1.5,
                         tactile_limb_sigma:       float = 3.0,
                         fill_holes:               bool  = False,
                         fill_holes_min_area:      int   = 20,
                         fill_holes_max_area:      int   = 2000,
                         fill_holes_kernel:        int   = 5,
                         **kwargs) -> None:
    """
    Convenience wrapper that calls run_pipeline() with tactile-safe defaults.

    Key differences from the visual pipeline:
      • detail_strength = 0      – no micro-texture (felt as noise, not shape)
      • mesh_px         = 140    – lower resolution → natural low-pass from
                                   the Lanczos downscale in depth_to_stl()
      • relief_mm       = 7.0    – shallower relief (tyflographic standard:
                                   contours don't 'jab' the fingertip)
      • clahe_clip      = 0.0    – local contrast enhancement meaningless for touch
      • smooth_sigma    = 0.0    – smoothing done by prepare_for_touch()
      • tactile         = True   – activates median + broad-Gaussian pipeline

    When tactile_levels > 1 the pipeline uses foreground-aware quantization
    (quantize_depth_foreground_aware) instead of the legacy equal-area method,
    followed by morphological boundary smoothing (smooth_quantized_boundaries).

    Args:
        input_path:                Path to the input image.
        output_dir:                Output directory.
        width_mm:                  Physical width [mm].
        relief_mm:                 Max relief height (default 7 mm for tactile).
        base_mm:                   Base plate thickness [mm].
        mesh_px:                   Mesh resolution (default 140 px for tactile).
        tactile_median_size:       Spike-removal kernel size (default 5).
        tactile_sigma:             Gaussian σ for tactile smoothing (default 3.5).
        tactile_levels:            Enable discrete levels when > 1.
                                   0 = continuous gradient.
        tactile_fg_threshold_pct:  Percentile splitting bg / fg (default 40.0).
        tactile_bg_levels:         Levels for background zone (default 2).
        tactile_fg_levels:         Levels for foreground/figure zone (default 4).
        tactile_boundary_kernel:   Morphological smoothing kernel size (default 9).
        tactile_multiscale:       bool  = False,
        tactile_fine_sigma:       float = 1.5,
        tactile_limb_sigma:       float = 3.0,
        **kwargs:                  Any other run_pipeline() keyword args.
    """
    return run_pipeline(
        input_path               = input_path,
        output_dir               = output_dir,
        width_mm                 = width_mm,
        relief_mm                = relief_mm,
        base_mm                  = base_mm,
        mesh_px                  = mesh_px,
        clahe_clip               = 0.0,
        smooth_sigma             = 0.0,
        detail_strength          = 0.0,
        tactile                  = True,
        tactile_median_size      = tactile_median_size,
        tactile_sigma            = tactile_sigma,
        tactile_levels           = tactile_levels,
        tactile_fg_threshold_pct = tactile_fg_threshold_pct,
        tactile_bg_levels        = tactile_bg_levels,
        tactile_fg_levels        = tactile_fg_levels,
        tactile_boundary_kernel  = tactile_boundary_kernel,
        tactile_multiscale       = tactile_multiscale,
        tactile_fine_sigma       = tactile_fine_sigma,
        tactile_limb_sigma       = tactile_limb_sigma,
        fill_holes               = fill_holes,
        fill_holes_min_area      = fill_holes_min_area,
        fill_holes_max_area      = fill_holes_max_area,
        fill_holes_kernel        = fill_holes_kernel,
        **kwargs,
    )


def _save_comparison(orig: np.ndarray, depth_maps: dict,
                     ensemble: np.ndarray,
                     out_dir: Path, stem: str) -> None:
    """Comparison grid 2×N: original + maps + ensemble (PNG, 2-column layout)."""
    PW = 1280
    PH = max(200, int(PW * orig.shape[0] / orig.shape[1]))
    PH = min(PH, 960)   # cap cell height to keep grid manageable

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

    out_path = out_dir / f"{stem}_comparison.png"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_PNG_COMPRESSION, 3])


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
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit (0 = disabled; 2.0 recovers fabric/face detail)")
    parser.add_argument("--smooth-sigma", type=float, default=0.7,
                        help="Gaussian σ [px] for final noise pass (guided filter already "
                             "smooths globally; keep low to preserve fine detail)")
    parser.add_argument("--invert-depth", action="store_true", default=False,
                        help="Invert depth values (convex↔concave)")
    parser.add_argument("--flip-x", action="store_true", default=False,
                        help="Horizontal mirror of STL")
    parser.add_argument("--no-flip-y", action="store_true", default=False,
                        help="Disable default vertical flip (flip_y=True corrects "
                             "image→Prusa Slicer orientation)")
    parser.add_argument("--detail-strength", type=float, default=0.15,
                        help="Micro-detail overlay amplitude (0 = disabled; "
                             "0.10–0.20 recommended; ±30%% of relief height max)")
    parser.add_argument("--detail-blur-sigma", type=float, default=1.2,
                        help="Low-pass cutoff for detail extraction [px] "
                             "(1.0 = fine grain/skin, 1.5 = broader folds/brushstrokes)")

    # ── Tactile / tyflographic mode ───────────────────────────────────────────
    tactile_grp = parser.add_argument_group(
        "Tactile / tyflographic output",
        "Settings for 3D prints intended to be read by touch (visually impaired).\n"
        "Use --tactile to activate; overrides detail/CLAHE/sigma automatically.",
    )
    tactile_grp.add_argument("--tactile", action="store_true", default=False,
                             help="Enable tyflographic mode: disables micro-detail, "
                                  "applies median+Gaussian smoothing, disables CLAHE. "
                                  "Recommended defaults: --mesh-px 140 --relief-mm 7")
    tactile_grp.add_argument("--tactile-median", type=int, default=5,
                             help="Median filter size for spike removal [px] "
                                  "(tactile mode only; odd integer, default 5)")
    tactile_grp.add_argument("--tactile-sigma", type=float, default=3.5,
                             help="Gaussian σ for broad tactile smoothing [px] "
                                  "(tactile mode only; 3.0–4.0 recommended)")
    tactile_grp.add_argument("--tactile-levels", type=int, default=0,
                             help="Quantize depth to N total levels using foreground-aware "
                                  "quantization (background + foreground zones). "
                                  "Set > 1 to activate; 0 = continuous gradient. "
                                  "Actual level count = --tactile-bg-levels + --tactile-fg-levels")
    tactile_grp.add_argument("--tactile-fg-threshold", type=float, default=40.0,
                             help="Percentile (0–100) splitting background from foreground "
                                  "in foreground-aware quantization (default 40.0). "
                                  "Lower values → more pixels treated as foreground.")
    tactile_grp.add_argument("--tactile-bg-levels", type=int, default=2,
                             help="Discrete height levels assigned to the background zone "
                                  "(default 2). Background = depth values below "
                                  "--tactile-fg-threshold percentile.")
    tactile_grp.add_argument("--tactile-fg-levels", type=int, default=4,
                             help="Discrete height levels assigned to the foreground zone "
                                  "(default 4). Foreground = figure / main subject.")
    tactile_grp.add_argument("--tactile-boundary-kernel", type=int, default=9,
                             help="Morphological kernel size (px) for smoothing jagged "
                                  "boundaries between quantization levels (default 9). "
                                  "Larger → smoother edges but may erode fine details.")
    tactile_grp.add_argument("--tactile-multiscale", action="store_true", default=False,
                             help="Use multi-scale smoothing instead of single-pass Gaussian. "
                                  "Preserves limb-scale contours (legs, arms) while removing "
                                  "fine texture noise (fabric folds, grass blades). "
                                  "Combine with --tactile-fine-sigma and --tactile-limb-sigma.")
    tactile_grp.add_argument("--tactile-fine-sigma", type=float, default=1.5,
                             help="σ [px] for fine-texture removal in multiscale mode "
                                  "(default 1.5). Removes fabric/grass noise below this scale. "
                                  "Recommended: 1.2–1.5.")
    tactile_grp.add_argument("--tactile-limb-sigma", type=float, default=3.0,
                             help="σ [px] defining limb-scale in multiscale mode "
                                  "(default 3.0). Final filter uses limb_sigma*0.5 to "
                                  "preserve separation between adjacent legs/arms. "
                                  "Recommended: 2.5–3.5.")
    tactile_grp.add_argument("--tactile-layered", action="store_true", default=False,
                             help="Layered relief for touch: subtract the ground, raise every "
                                  "object as its own plane with a palpable wall. Implies "
                                  "--tactile and replaces --fill-holes / --detail-* / "
                                  "--tactile-multiscale / --tactile-levels. Recommended for "
                                  "scenes where figures stand on a receding ground.")
    tactile_grp.add_argument("--plane-mask", action="append", default=None, metavar="N=FILE",
                             help="Plane mask for --tactile-layered (repeatable). N: 1 = "
                                  "foreground (highest), 2 = middle ground, … White = inside. "
                                  "Without masks objects are segmented automatically.")
    tactile_grp.add_argument("--layered-threshold", type=float, default=0.035,
                             help="How far above the ground a region must rise to count as an "
                                  "object (fraction of depth range). Lower = more, smaller objects.")
    tactile_grp.add_argument("--layered-detail", type=float, default=0.08,
                             help="Luminance detail inside objects only (folds, limb contours); "
                                  "0 = smooth planes.")
    tactile_grp.add_argument("--layered-edge", type=float, default=5.0,
                             help="Width in mm of the slope where an object meets the ground, "
                                  "centred on the contour; 0 = vertical wall.")
    tactile_grp.add_argument("--layered-texture", type=float, default=1.0,
                             help="Background textures: grass roughness growing towards the "
                                  "viewer, raised lines for sticks and stalks, soft clouds. "
                                  "Multiplier; 0 = smooth ground and sky.")
    parser.add_argument("--stl-smooth", type=float, default=0.03,
                        help="Edge-preserving smoothing of the STL surface (fraction of relief; "
                             "smooths undulation below this height, keeps walls; 0 = off)")

    # ── Small-object hole filling ─────────────────────────────────────────────
    holes_grp = parser.add_argument_group(
        "Small-object hole filling",
        "Fills flat/empty interiors of small objects (animals, distant figures)\n"
        "whose depth is not detected by DPT/MiDaS.  Applied right after fusion.",
    )
    holes_grp.add_argument("--fill-holes", action="store_true", default=False,
                           help="Enable small-object hole filling after ensemble fusion.")
    holes_grp.add_argument("--fill-holes-min-area", type=int, default=20,
                           help="Minimum contour area [px²] to fill (default 20).")
    holes_grp.add_argument("--fill-holes-max-area", type=int, default=2000,
                           help="Maximum contour area [px²] to fill (default 2000). "
                                "Tune to the approximate pixel area of the small object.")
    holes_grp.add_argument("--fill-holes-kernel", type=int, default=5,
                           help="Morphological kernel size for contour closing (default 5).")

    args = parser.parse_args()

    # When --tactile is set without explicit mesh-px/relief-mm, apply tactile defaults
    mesh_px   = args.mesh_px
    relief_mm = args.relief_mm
    # Sentinel comparison: argparse cannot tell "passed 512" from "passed nothing".
    if args.tactile_layered:
        # Planes need crisp walls: 0.5 mm cells at 200 mm.  The low 140 px grid
        # of the continuous tactile mode was a low-pass for noise this mode no
        # longer has.
        if mesh_px   == 512:   mesh_px   = 400
        if relief_mm == 10.0:  relief_mm = 7.0
    elif args.tactile:
        if mesh_px   == 512:   mesh_px   = 140
        if relief_mm == 10.0:  relief_mm = 7.0

    run_pipeline(
        input_path               = args.input,
        output_dir               = args.output_dir,
        width_mm                 = args.width_mm,
        relief_mm                = relief_mm,
        base_mm                  = args.base_mm,
        mesh_px                  = mesh_px,
        clahe_clip               = args.clahe_clip,
        smooth_sigma             = args.smooth_sigma,
        invert_depth             = args.invert_depth,
        flip_x                   = args.flip_x,
        flip_y                   = not args.no_flip_y,
        detail_strength          = args.detail_strength,
        detail_blur_sigma        = args.detail_blur_sigma,
        tactile                  = args.tactile,
        tactile_median_size      = args.tactile_median,
        tactile_sigma            = args.tactile_sigma,
        tactile_levels           = args.tactile_levels,
        tactile_fg_threshold_pct = args.tactile_fg_threshold,
        tactile_bg_levels        = args.tactile_bg_levels,
        tactile_fg_levels        = args.tactile_fg_levels,
        tactile_boundary_kernel  = args.tactile_boundary_kernel,
        tactile_multiscale       = args.tactile_multiscale,
        tactile_fine_sigma       = args.tactile_fine_sigma,
        tactile_limb_sigma       = args.tactile_limb_sigma,
        fill_holes               = args.fill_holes,
        fill_holes_min_area      = args.fill_holes_min_area,
        fill_holes_max_area      = args.fill_holes_max_area,
        fill_holes_kernel        = args.fill_holes_kernel,
        tactile_layered          = args.tactile_layered,
        plane_masks              = args.plane_mask,
        layered_threshold        = args.layered_threshold,
        layered_detail           = args.layered_detail,
        layered_texture          = args.layered_texture,
        layered_edge_mm          = args.layered_edge,
        stl_smooth               = args.stl_smooth,
    )


if __name__ == "__main__":
    main()

