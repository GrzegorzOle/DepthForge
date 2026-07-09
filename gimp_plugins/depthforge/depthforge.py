#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ── EARLIEST POSSIBLE LOG – runs before any import ────────────────────────────
import sys as _sys, os as _os, tempfile as _tmp, datetime as _dt
try:
    _el = _os.path.join(_tmp.gettempdir(), "depthforge_early.log")
    with open(_el, "a", encoding="utf-8") as _f:
        _f.write(f"[{_dt.datetime.now():%H:%M:%S}] LOAD argv={_sys.argv}\n")
        _f.write(f"[{_dt.datetime.now():%H:%M:%S}]      file={__file__}  cwd={_os.getcwd()}\n")
except Exception as _e:
    pass   # silently ignore – we try again after gi imports
# ──────────────────────────────────────────────────────────────────────────────
"""
DepthForge – GIMP 3.x Plugin  (GIMP 3.2.x)

Architecture
------------
GIMP bundles Python 3.14 which has no binary wheels for numpy/opencv.
This plugin calls an EXTERNAL Python interpreter (project .venv) via
subprocess to generate the depth map, then loads the result back into GIMP.

Menu entries
------------
  Filters > DepthForge > Generate Depth Map…
  Filters > DepthForge > Diagnose (test configuration)
"""

import sys
import os
import struct
import zlib
import tempfile
import subprocess
import traceback
import json as _json

import gi
gi.require_version("Gimp",   "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk",    "3.0")
gi.require_version("Gegl",   "0.4")

from gi.repository import Gimp, GimpUi, GLib, GObject, Gio, Gegl

PROC_NAME  = "plug-in-depthforge"
PROC_DIAG  = "plug-in-depthforge-diagnose"
MENU_PATH  = "<Image>/Filters/DepthForge"

# Resolve plugin directory robustly:
# In GIMP 3.x, __file__ may be a bare filename with no directory.
# sys.argv[0] is always the full absolute path to the script.
_PLUGIN_SCRIPT = os.path.realpath(sys.argv[0] if sys.argv else __file__)
_PLUGIN_DIR    = os.path.dirname(_PLUGIN_SCRIPT)

# Log to 3 locations to guarantee at least one is writable
_LOG_FILE  = os.path.join(_PLUGIN_DIR,           "depthforge_run.log")
_LOG_TEMP  = os.path.join(tempfile.gettempdir(),  "depthforge_run.log")
# Fallback: hardcoded project path (set by install.py)
_LOG_PROJ  = ""   # filled from install JSON below


def _log(msg: str):
    import datetime
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n"
    for path in (_LOG_FILE, _LOG_TEMP, _LOG_PROJ):
        if not path:
            continue
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            break           # stop after first successful write
        except Exception:
            pass


# Try to load project root from install JSON (written by install.py)
try:
    _cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(_cfg_path):
        import json as _j
        _d = _j.load(open(_cfg_path, encoding="utf-8"))
        _root = _d.get("project_root", "")
        if _root:
            _LOG_PROJ = os.path.join(_root, "depthforge_gimp.log")
except Exception:
    pass


# Write startup entry immediately when GIMP loads the plugin
_log(f"=== Plugin module loaded | argv0={sys.argv[0] if sys.argv else '?'}")
_log(f"    __file__   = {__file__}")
_log(f"    PLUGIN_DIR = {_PLUGIN_DIR}")
_log(f"    LOG_FILE   = {_LOG_FILE}")
_log(f"    LOG_TEMP   = {_LOG_TEMP}")
_log(f"    LOG_PROJ   = {_LOG_PROJ}")


# ─────────────────────────────────────────────────────────────────────────────
#  Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_project_root() -> str | None:
    """
    Find the DepthForge project root (directory that contains config.json).
    Priority:
      1. depthforge_install.json (always correct – written by install.py)
      2. Walk up from plugin directory looking for config.json
    """
    # 1. Install JSON – always has the correct path
    cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = _json.load(f)
            root = data.get("project_root", "").strip()
            if root and os.path.isfile(os.path.join(root, "config.json")):
                return root
        except Exception:
            pass

    # 2. Walk up from plugin directory
    here = _PLUGIN_DIR
    for _ in range(10):
        if os.path.isfile(os.path.join(here, "config.json")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _find_external_python() -> str | None:
    """
    Priority:
      1. DEPTHFORGE_PYTHON environment variable
      2. depthforge_install.json  (written by install.py)
      3. Auto-discovery of project .venv
      4. System Python
    """
    # 1. Env var
    env = os.environ.get("DEPTHFORGE_PYTHON", "").strip()
    if env and os.path.isfile(env):
        _log(f"Python found via env var: {env}")
        return env

    # 2. Install config JSON
    cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = _json.load(f)
            venv_py = data.get("venv_python", "").strip()
            if venv_py and os.path.isfile(venv_py):
                _log(f"Python found via install JSON: {venv_py}")
                return venv_py
        except Exception as e:
            _log(f"Could not read install JSON: {e}")

    # 3. Auto-discovery
    root = _find_project_root()
    if root:
        for rel in (
            os.path.join(".venv", "Scripts", "python.exe"),
            os.path.join(".venv", "bin",     "python3"),
            os.path.join(".venv", "bin",     "python"),
            os.path.join("venv",  "Scripts", "python.exe"),
            os.path.join("venv",  "bin",     "python3"),
        ):
            cand = os.path.join(root, rel)
            if os.path.isfile(cand):
                _log(f"Python found via auto-discovery: {cand}")
                return cand

    # 4. System fallback
    import shutil
    py = shutil.which("python3") or shutil.which("python")
    if py:
        _log(f"Python found via PATH: {py}")
    return py


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal PNG writer  (pure stdlib – struct + zlib)
# ─────────────────────────────────────────────────────────────────────────────

def _write_png_rgb(path: str, width: int, height: int, rgb_bytes: bytes):
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    stride   = width * 3
    raw_rows = b"".join(b"\x00" + rgb_bytes[y * stride:(y + 1) * stride]
                        for y in range(height))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw_rows, 6))
                 + chunk(b"IEND", b""))


# ─────────────────────────────────────────────────────────────────────────────
#  External helper script  (runs inside project .venv)
# ─────────────────────────────────────────────────────────────────────────────

_HELPER_SCRIPT = r"""
# DepthForge GIMP helper – runs in project .venv (has numpy, opencv, openvino)
# Pipeline: Standard + OpenVINO MiDaS + OpenVINO DPT → ensemble fusion →
#   Visual:  postprocess_depth (CLAHE + Gaussian)
#   Tactile: fill_holes → detail_overlay → multiscale_smooth (v9 params)
import sys, os

input_png      = sys.argv[1]
output_png     = sys.argv[2]
enhancement    = int(sys.argv[3])              if len(sys.argv) > 3 else 50
invert         = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
project_root   = sys.argv[5]                   if len(sys.argv) > 5 else ""
color_out_png  = sys.argv[6]                   if len(sys.argv) > 6 else ""
stl_out_path   = sys.argv[7]                   if len(sys.argv) > 7 else ""
tactile_mode   = sys.argv[8].lower() == "true" if len(sys.argv) > 8 else False

# ── Make DepthForge importable ────────────────────────────────────────────────
if project_root:
    for p in (os.path.join(project_root, "src"), project_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

try:
    import numpy as np
    import cv2
except ImportError as e:
    sys.exit(f"IMPORT_ERROR: {e}")

img = cv2.imread(input_png)
if img is None:
    sys.exit(f"READ_ERROR: cannot read {input_png}")

# enhancement (0-100) → clahe_clip dla postprocess_depth (tylko tryb visual)
# domyślna wartość w pipeline = 2.0 odpowiada enhancement=50
clahe_clip = enhancement / 25.0   # 0→0.0, 50→2.0, 100→4.0

depth_u8 = None   # wynik końcowy uint8 [0-255]

# ── 1. Pełny pipeline DepthForge: Standard + MiDaS OV + DPT OV + ensemble ────
try:
    from depth_pipeline import (
        fuse_depth_maps, postprocess_depth,
        normalize_f32_robust, DepthForge,
        fill_small_object_holes, apply_detail_overlay,
        prepare_for_touch_multiscale,
    )
    import time

    # WAŻNE: config.json używa ścieżek względnych – ustawiamy CWD na project_root
    if project_root and os.path.isdir(project_root):
        os.chdir(project_root)
        print(f"CWD set to project_root: {project_root}", flush=True)
    cfg_path = os.path.join(project_root, "config.json") if project_root else "config.json"
    df = DepthForge(cfg_path if os.path.isfile(cfg_path) else None)
    de_cfg = df.config.get("model", {}).get("depth_estimation", {})

    midas_compiled = None
    dpt_compiled   = None
    if df.ov_core:
        midas_path = de_cfg.get("midas_model_path")
        if midas_path and os.path.isfile(midas_path):
            midas_compiled = df._load_openvino_model(midas_path)
            print(f"OpenVINO MiDaS loaded: {midas_path}", flush=True)
        else:
            print(f"OpenVINO MiDaS NOT found: {midas_path}", flush=True)
        dpt_path = de_cfg.get("dpt_model_path")
        if dpt_path and os.path.isfile(dpt_path):
            dpt_compiled = df._load_openvino_model(dpt_path)
            print(f"OpenVINO DPT loaded:   {dpt_path}", flush=True)
        else:
            print(f"OpenVINO DPT NOT found: {dpt_path}", flush=True)

    depth_maps_f32 = {}

    # Standard (syntetyczny)
    t0 = time.perf_counter()
    s = df.generate_depth_map_midas(img)
    if s is not None:
        if s.ndim == 3:
            s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        depth_maps_f32["standard"] = normalize_f32_robust(s.astype(np.float32))
        print(f"  Standard:  {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    # MiDaS OpenVINO
    if midas_compiled:
        t0 = time.perf_counter()
        m = df._run_openvino_inference_f32(midas_compiled, img, "midas")
        if m is not None:
            depth_maps_f32["midas"] = m.astype(np.float32)
            print(f"  MiDaS OV:  {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    # DPT OpenVINO
    if dpt_compiled:
        t0 = time.perf_counter()
        d = df._run_openvino_inference_f32(dpt_compiled, img, "dpt")
        if d is not None:
            depth_maps_f32["dpt"] = d.astype(np.float32)
            print(f"  DPT OV:    {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    print(f"  Modele: {list(depth_maps_f32.keys())}", flush=True)

    if not depth_maps_f32:
        raise RuntimeError("Żaden model nie zwrócił mapy głębi")

    # Fuzja ensemble (scale-shift alignment)
    if len(depth_maps_f32) > 1:
        fused = fuse_depth_maps(depth_maps_f32)
        print(f"  Ensemble fusion: {list(depth_maps_f32.keys())}", flush=True)
    else:
        key = list(depth_maps_f32.keys())[0]
        fused = normalize_f32_robust(list(depth_maps_f32.values())[0])
        print(f"  Pojedynczy model: {key}", flush=True)

    # ── Post-processing ───────────────────────────────────────────────────────
    if tactile_mode:
        # === TRYB TAKTYLNY (parametry v9) =====================================
        # Identyczny pipeline jak indian_summer_tactile_v9:
        #   fill_holes → detail_overlay(0.05, 2.5) → multiscale_smooth(1.5, 3.0)
        #   → postprocess bez CLAHE i Gaussiana (obsługiwane przez multiscale)
        print("  Tryb: TAKTYLNY (v9 params)", flush=True)

        # Krok 1: wypełnienie wnętrz małych obiektów
        fused = fill_small_object_holes(fused, min_area=20, max_area=2000, kernel_size=5)
        print("  fill_holes OK", flush=True)

        # Krok 2: nakładka mikrodetali (kontury kończyn z luminancji)
        fused = apply_detail_overlay(fused, img, strength=0.05, blur_sigma=2.5)
        print("  detail_overlay(0.05, 2.5) OK", flush=True)

        # Krok 3: wieloskalowe wygładzanie taktylne
        fused = prepare_for_touch_multiscale(
            fused, median_size=5, fine_sigma=1.5, limb_sigma=3.0
        )
        print("  multiscale_smooth(fine=1.5, limb=3.0) OK", flush=True)

        # Krok 4: postprocess bez CLAHE (prepare_for_touch już wygłodziło)
        depth_u8 = postprocess_depth(fused, clahe_clip=0.0, sigma=0.0)
    else:
        # === TRYB WIZUALNY (standardowy) ======================================
        print("  Tryb: WIZUALNY (CLAHE + Gaussian)", flush=True)
        depth_u8 = postprocess_depth(fused, clahe_clip=clahe_clip, sigma=0.7)

    print("  Pipeline OK", flush=True)

except Exception as e:
    import traceback
    print(f"Pipeline error (fallback do syntetycznego): {e}", flush=True)
    traceback.print_exc()
    depth_u8 = None

# ── 2. Fallback: syntetyczny estymator jeśli pipeline się nie powiódł ─────────
if depth_u8 is None:
    print("Fallback: syntetyczny estymator głębi", flush=True)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    def _n(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-6)
    fused_fb = _n(255.0 - gray) * 0.70 + _n(edges) * 0.30
    fb_u8 = (fused_fb * 255).clip(0, 255).astype(np.uint8)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        fb_u8 = clahe.apply(fb_u8)
    depth_u8 = fb_u8

# ── 3. Inwersja + dopasowanie rozmiaru ───────────────────────────────────────
if invert:
    depth_u8 = 255 - depth_u8

h, w = img.shape[:2]
if depth_u8.shape[:2] != (h, w):
    depth_u8 = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_LINEAR)

ok = cv2.imwrite(output_png, depth_u8)
if not ok:
    sys.exit(f"WRITE_ERROR: cannot write {output_png}")
print(f"OK:{output_png}", flush=True)

# ── 4. Kolorowa mapa głębi (INFERNO) ─────────────────────────────────────────
if color_out_png:
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    ok_c = cv2.imwrite(color_out_png, depth_color)
    if ok_c:
        print(f"COLOR_OK:{color_out_png}", flush=True)
    else:
        print(f"COLOR_FAIL:{color_out_png}", flush=True)

# ── 5. Eksport STL ────────────────────────────────────────────────────────────
if stl_out_path:
    try:
        from depth_pipeline import depth_to_stl
        depth_to_stl(depth_u8, stl_out_path)
        print(f"STL_OK:{stl_out_path}", flush=True)
    except Exception as e:
        import traceback
        print(f"STL_FAIL:{e}", flush=True)
        traceback.print_exc()
"""



# ─────────────────────────────────────────────────────────────────────────────
#  GIMP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_error(procedure, message: str):
    _log(f"ERROR returned to GIMP: {message[:200]}")
    return procedure.new_return_values(
        Gimp.PDBStatusType.EXECUTION_ERROR,
        GLib.Error.new_literal(GLib.quark_from_string("depthforge"), message, 0),
    )


def _load_result_image(path: str):
    """Load PNG file as a new GIMP image (GIMP 3.x compatible)."""
    gfile = Gio.File.new_for_path(path)
    # Try the simplest form first (GIMP 3.x)
    try:
        return Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, gfile)
    except TypeError:
        pass
    # Some builds need explicit extra args – try via PDB
    try:
        pdb    = Gimp.get_pdb()
        proc   = pdb.lookup_procedure("file-png-load")
        cfg    = proc.create_config()
        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
        cfg.set_property("file", gfile)
        result = proc.run(cfg)
        return result.index(1)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin class
# ─────────────────────────────────────────────────────────────────────────────

class DepthForgePlugin(Gimp.PlugIn):

    def do_query_procedures(self):
        return [PROC_NAME, PROC_DIAG]

    def do_set_i18n(self, procname):
        return False, None, None

    # ── Create procedures ─────────────────────────────────────────────────────
    def do_create_procedure(self, name):
        if name == PROC_DIAG:
            return self._create_diag_procedure(name)
        return self._create_main_procedure(name)

    def _create_main_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Generate Depth Map\u2026")
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "DepthForge \u2013 Depth Map Generator",
            "Generates a depth map from the active layer.",
            name,
        )
        procedure.set_attribution("DepthForge Team", "DepthForge Team", "2026")
        procedure.add_int_argument(
            "enhancement-level", "_Enhancement (0-100)",
            "CLAHE contrast enhancement (0=none, 100=max).",
            0, 100, 50, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "invert-depth", "_Invert depth",
            "Invert depth polarity.", False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "color-depth-layer", "Warstwa _kolorowa (INFERNO)",
            "Zaznacz = warstwa głębi w kolorach INFERNO. "
            "Odznacz = warstwa głębi czarno-biała.",
            False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "export-stl", "Export _STL (3D model)",
            "Export depth map as a watertight STL file for 3D printing.",
            False, GObject.ParamFlags.READWRITE,
        )
        procedure.add_boolean_argument(
            "tactile-mode", "_Tryb taktylny (v9 – fill-holes + detail + multiscale)",
            "Włącza pełny pipeline taktylny (jak indian_summer_tactile_v9): "
            "fill_holes → detail_overlay(0.05,2.5) → multiscale_smooth(1.5,3.0). "
            "Wyłącz dla trybu wizualnego z CLAHE.",
            False, GObject.ParamFlags.READWRITE,
        )
        return procedure

    def _create_diag_procedure(self, name):
        procedure = Gimp.Procedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run_diag, None
        )
        procedure.set_menu_label("Diagnose\u2026")
        procedure.add_menu_path(MENU_PATH)
        procedure.set_documentation(
            "DepthForge \u2013 Diagnose",
            "Tests the DepthForge configuration and shows a status report.",
            name,
        )
        procedure.set_attribution("DepthForge Team", "DepthForge Team", "2026")
        return procedure

    # ── Diagnose ──────────────────────────────────────────────────────────────
    def run_diag(self, procedure, run_mode, args, run_data):
        GimpUi.init("depthforge-diag")
        lines = ["DepthForge – Diagnostic Report", "=" * 40]

        ext_py = _find_external_python()
        if ext_py:
            lines.append(f"[OK] External Python:\n     {ext_py}")
        else:
            lines.append("[FAIL] External Python: NOT FOUND")
            lines.append("       Run install.py or set DEPTHFORGE_PYTHON env var")

        cfg_path = os.path.join(_PLUGIN_DIR, "depthforge_install.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    data = _json.load(f)
                lines.append(f"[OK] Install config: {cfg_path}")
                lines.append(f"     project_root = {data.get('project_root','?')}")
                lines.append(f"     venv_python  = {data.get('venv_python','?')}")
            except Exception as e:
                lines.append(f"[WARN] Install config unreadable: {e}")
        else:
            lines.append(f"[WARN] depthforge_install.json missing:\n     {cfg_path}")
            lines.append("       Run install.py to create it")

        root = _find_project_root()
        lines.append(f"{'[OK]' if root else '[WARN]'} Project root: {root or 'not found'}")

        if ext_py:
            r = subprocess.run(
                [ext_py, "-c",
                 "import numpy, cv2; print(f'numpy {numpy.__version__}, "
                 "opencv {cv2.__version__}')"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                lines.append(f"[OK] Python deps: {r.stdout.strip()}")
            else:
                lines.append(f"[FAIL] Python deps:\n{r.stderr.strip()[:300]}")

        env_val = os.environ.get("DEPTHFORGE_PYTHON", "")
        lines.append(f"{'[OK]' if env_val else '[INFO]'} DEPTHFORGE_PYTHON env = "
                     f"{env_val or '(not set – using install JSON)'}")

        lines.append("=" * 40)
        lines.append(f"Log file: {_LOG_FILE}")

        msg = "\n".join(lines)
        _log("Diagnose ran:\n" + msg)

        dialog = Gtk = None
        try:
            from gi.repository import Gtk as _Gtk
            Gtk = _Gtk
            dialog = Gtk.MessageDialog(
                message_type=Gtk.MessageType.INFO,
                buttons=Gtk.ButtonsType.OK,
                text="DepthForge Diagnostic",
            )
            dialog.format_secondary_text(msg)
            dialog.run()
            dialog.destroy()
        except Exception:
            pass

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self, procedure, run_mode, image, drawables, config, run_data):
        _log(f"run() called, run_mode={run_mode}")

        enhancement  = config.get_property("enhancement-level")
        invert_depth = config.get_property("invert-depth")

        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("depthforge")
            dialog = GimpUi.ProcedureDialog.new(
                procedure, config, "DepthForge \u2013 Depth Map Generator"
            )
            dialog.fill(None)
            ok = dialog.run()
            dialog.destroy()
            if not ok:
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )
            enhancement  = config.get_property("enhancement-level")
            invert_depth = config.get_property("invert-depth")

        color_depth_layer = config.get_property("color-depth-layer")
        export_stl        = config.get_property("export-stl")
        tactile_mode      = config.get_property("tactile-mode")

        _log(f"params: enhancement={enhancement} invert={invert_depth} "
             f"color={color_depth_layer} stl={export_stl} tactile={tactile_mode}")

        ext_python = _find_external_python()
        if not ext_python:
            return _make_error(
                procedure,
                "DepthForge: nie znaleziono zewnetrznego Pythona z numpy/opencv.\n\n"
                "Uruchom w PowerShell:\n"
                "  python H:\\test\\DepthForge\\install.py --yes\n\n"
                "Lub ustaw zmienna srodowiskowa:\n"
                "  DEPTHFORGE_PYTHON = sciezka\\do\\python.exe\n\n"
                f"(szukano w: {_PLUGIN_DIR}\\depthforge_install.json)\n"
                f"Log: {_LOG_FILE}",
            )

        Gimp.context_push()
        image.undo_group_start()
        tmp_in = tmp_out = tmp_helper = tmp_color = None

        # Duplikat obrazu do eksportu (spłaszczony – wszystkie warstwy złożone)
        export_image = None
        try:
            drawable = drawables[0] if drawables else image.get_active_drawable()
            w = image.get_width()
            h = image.get_height()
            _log(f"Image size: {w}x{h}, base_type: {image.get_base_type()}, "
                 f"layers: {len(image.get_layers())}")

            # Tworzymy spłaszczoną kopię całego obrazu (wszystkie widoczne warstwy złożone).
            # Dzięki temu model głębi widzi dokładnie to samo co widzisz na ekranie,
            # a nie tylko wybraną warstwę.
            try:
                export_image = image.duplicate()
                export_image.flatten()
                export_layers = export_image.get_layers()
                export_drawable = export_layers[0] if export_layers else export_image.get_active_drawable()
                _log("Using flattened duplicate for depth estimation (full composite)")
            except Exception as e:
                _log(f"Could not flatten duplicate ({e}) – falling back to active drawable")
                export_image = None
                export_drawable = drawable

            # Konwertuj do RGB jeśli obraz jest w trybie GRAY lub ma kanał alfa
            # (modele OpenVINO oczekują BGR/RGB 3-kanałowego wejścia)
            if export_image is not None:
                try:
                    base = export_image.get_base_type()
                    # Gimp.ImageBaseType.GRAY == 1
                    if int(base) != 0:   # 0 = RGB
                        pdb_conv = Gimp.get_pdb()
                        proc_conv = pdb_conv.lookup_procedure("gimp-image-convert-rgb")
                        if proc_conv:
                            cfg_conv = proc_conv.create_config()
                            cfg_conv.set_property("image", export_image)
                            proc_conv.run(cfg_conv)
                            export_layers = export_image.get_layers()
                            export_drawable = export_layers[0] if export_layers else export_image.get_active_drawable()
                            _log("Converted export image to RGB")
                except Exception as e:
                    _log(f"RGB conversion warning: {e}")

            # 1. Save image to temp PNG – GIMP 3.2.x multi-method export
            tmp_fd, tmp_in = tempfile.mkstemp(suffix=".png", prefix="df_in_")
            os.close(tmp_fd)
            saved = False
            pdb = Gimp.get_pdb()
            src_image   = export_image if export_image is not None else image
            src_drawable = export_drawable

            # Method A: file-png-export PDB (confirmed available in GIMP 3.2.4)
            if not saved:
                try:
                    proc = pdb.lookup_procedure("file-png-export")
                    if proc:
                        cfg = proc.create_config()
                        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
                        cfg.set_property("image",    src_image)
                        cfg.set_property("file",     Gio.File.new_for_path(tmp_in))
                        for prop in ("drawables", "drawable"):
                            try:
                                val = [src_drawable] if prop == "drawables" else src_drawable
                                cfg.set_property(prop, val)
                                break
                            except Exception:
                                pass
                        proc.run(cfg)
                        if os.path.isfile(tmp_in) and os.path.getsize(tmp_in) > 100:
                            saved = True
                            _log(f"Saved via file-png-export: {tmp_in}")
                except Exception as e:
                    _log(f"file-png-export failed: {e}")

            # Method B: Gimp.file_save
            if not saved:
                for args in (
                    (Gimp.RunMode.NONINTERACTIVE, src_image, src_drawable,
                     Gio.File.new_for_path(tmp_in)),
                    (Gimp.RunMode.NONINTERACTIVE, src_image,
                     Gio.File.new_for_path(tmp_in)),
                ):
                    try:
                        Gimp.file_save(*args)
                        if os.path.isfile(tmp_in) and os.path.getsize(tmp_in) > 100:
                            saved = True
                            _log(f"Saved via Gimp.file_save({len(args)} args): {tmp_in}")
                            break
                    except Exception as e:
                        _log(f"  file_save {len(args)} args: {e}")

            # Method C: pixel-by-pixel via get_pixel → pure-Python PNG writer
            if not saved:
                try:
                    _log("Falling back to pixel-by-pixel read (slow for large images)")
                    buf = bytearray(w * h * 3)
                    for y in range(h):
                        for x in range(w):
                            px = src_drawable.get_pixel(x, y)
                            data = px[1] if isinstance(px, tuple) else px
                            r = data[0] if len(data) > 0 else 0
                            g = data[1] if len(data) > 1 else r
                            b = data[2] if len(data) > 2 else r
                            off = (y * w + x) * 3
                            buf[off] = r; buf[off+1] = g; buf[off+2] = b
                    _write_png_rgb(tmp_in, w, h, bytes(buf))
                    if os.path.getsize(tmp_in) > 100:
                        saved = True
                        _log(f"Saved via pixel-by-pixel: {tmp_in}")
                except Exception as e:
                    _log(f"pixel-by-pixel failed: {e}")

            if not saved:
                raise RuntimeError(
                    "Nie mozna eksportowac obrazu do pliku tymczasowego.\n"
                    f"Sprawdz log: {_LOG_FILE}"
                )
            _log(f"Input PNG size: {os.path.getsize(tmp_in)} bytes")

            tmp_fd, tmp_out = tempfile.mkstemp(suffix=".png", prefix="df_out_")
            os.close(tmp_fd)

            # Zawsze tworzymy ścieżkę tmp_color – helper zawsze generuje wersję kolorową.
            # Wybór która trafi do GIMP zależy od color_depth_layer.
            tmp_color = ""
            fd_c, tmp_color = tempfile.mkstemp(suffix=".png", prefix="df_color_")
            os.close(fd_c)

            stl_out_path = ""
            if export_stl:
                # Try to save STL alongside the image, otherwise use temp dir
                try:
                    img_uri = image.get_uri()
                    if img_uri:
                        import urllib.request
                        img_path = urllib.request.url2pathname(
                            img_uri.replace("file://", "").replace("file:///", "/")
                        )
                        base_no_ext = os.path.splitext(img_path)[0]
                        stl_out_path = base_no_ext + "_depth.stl"
                    else:
                        raise ValueError("no URI")
                except Exception:
                    fd_s, stl_out_path = tempfile.mkstemp(suffix=".stl", prefix="df_stl_")
                    os.close(fd_s)
                _log(f"STL output path: {stl_out_path}")

            # 2. Write helper script
            tmp_fd, tmp_helper = tempfile.mkstemp(suffix=".py", prefix="df_helper_")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(_HELPER_SCRIPT)

            # 3. Call external Python
            project_root = _find_project_root() or ""
            cmd = [
                ext_python, tmp_helper,
                tmp_in, tmp_out,
                str(enhancement),
                str(invert_depth).lower(),
                project_root,
                tmp_color,
                stl_out_path,
                str(tactile_mode).lower(),
            ]
            _log(f"Subprocess cmd: {' '.join(cmd[:3])} ...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            _log(f"Subprocess exit={result.returncode} stdout={result.stdout[:200]} "
                 f"stderr={result.stderr[:300]}")

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "(no output)").strip()
                raise RuntimeError(
                    f"Zewnetrzny Python zakonczyl sie bledem (kod {result.returncode}):\n\n"
                    f"{detail[:600]}"
                )

            if not os.path.isfile(tmp_out) or os.path.getsize(tmp_out) == 0:
                raise RuntimeError(
                    f"Plik wynikowy nie istnieje lub jest pusty:\n{tmp_out}\n\n"
                    f"stdout: {result.stdout[:200]}"
                )

            # 5. Załaduj wynik do GIMP jako nową warstwę
            _log(f"Loading result: {tmp_out} (color_mode={color_depth_layer})")

            # Wybierz plik źródłowy: kolorowy (INFERNO) lub czarno-biały
            if color_depth_layer and tmp_color and os.path.isfile(tmp_color) \
                    and os.path.getsize(tmp_color) > 0:
                src_result = tmp_color
                layer_name = "Depth Map – INFERNO (DepthForge)"
                _log("Using COLOR (INFERNO) result")
            else:
                src_result = tmp_out
                layer_name = "Depth Map (DepthForge)"
                _log("Using GRAYSCALE result")

            result_img = _load_result_image(src_result)
            if result_img is None:
                raise RuntimeError(f"Nie mozna zaladowac pliku wynikowego: {src_result}")

            layers = result_img.get_layers()
            result_drawable = layers[0] if layers else result_img.get_active_item()
            new_layer = Gimp.Layer.new_from_drawable(result_drawable, image)
            new_layer.set_name(layer_name)
            image.insert_layer(new_layer, None, -1)
            result_img.delete()

            Gimp.displays_flush()
            _log(f"Done – layer '{layer_name}' added")

            # Powiadomienie o eksporcie STL
            if export_stl and stl_out_path and os.path.isfile(stl_out_path):
                _log(f"STL exported: {stl_out_path}")
                try:
                    from gi.repository import Gtk as _Gtk
                    stl_folder = os.path.dirname(stl_out_path)
                    stl_name   = os.path.basename(stl_out_path)

                    _dlg = _Gtk.MessageDialog(
                        message_type=_Gtk.MessageType.INFO,
                        buttons=_Gtk.ButtonsType.NONE,
                        text="DepthForge – Eksport STL zakończony",
                    )
                    _dlg.format_secondary_text(
                        f"Model 3D zapisany:\n{stl_out_path}"
                    )
                    # Przyciski: Otwórz folder  +  OK
                    _dlg.add_button("Otwórz folder", 1)
                    _dlg.add_button("OK",            0)
                    _dlg.set_default_response(0)

                    response = _dlg.run()
                    _dlg.destroy()

                    if response == 1:
                        # Otwórz folder w eksploratorze i zaznacz plik STL
                        import subprocess as _sp, platform as _pl
                        try:
                            if _pl.system() == "Windows":
                                # /select, zaznacza plik w Eksploratorze
                                _sp.Popen(["explorer", "/select,", stl_out_path])
                            elif _pl.system() == "Darwin":
                                _sp.Popen(["open", "-R", stl_out_path])
                            else:
                                _sp.Popen(["xdg-open", stl_folder])
                        except Exception as _oe:
                            _log(f"open folder failed: {_oe}")
                except Exception as _de:
                    _log(f"STL dialog error: {_de}")

        except Exception:
            tb = traceback.format_exc()
            _log(f"EXCEPTION:\n{tb}")
            image.undo_group_end()
            Gimp.context_pop()
            return _make_error(procedure,
                               f"DepthForge blad:\n\n{tb[:800]}\n\nLog: {_LOG_FILE}")
        finally:
            # Usuń tymczasowe pliki
            for tmp in (tmp_in, tmp_out, tmp_helper, tmp_color):
                try:
                    if tmp and os.path.isfile(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass
            # Usuń spłaszczony duplikat obrazu (żeby nie zaśmiecać listy obrazów GIMP)
            try:
                if export_image is not None:
                    export_image.delete()
            except Exception:
                pass

        image.undo_group_end()
        Gimp.context_pop()
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


# ─────────────────────────────────────────────────────────────────────────────
Gimp.main(DepthForgePlugin.__gtype__, sys.argv)

