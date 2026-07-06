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
                 fill_holes_kernel:        int   = 5) -> None:
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
    if fill_holes:
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
    if tactile:
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
    if args.tactile:
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
    )


if __name__ == "__main__":
    main()

