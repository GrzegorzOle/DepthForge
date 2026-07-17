#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – bundle installer
=============================
Runs under the CPython interpreter shipped inside this bundle, never under a
system Python and never under GIMP's. It:

  1. finds GIMP's user plug-ins directory,
  2. copies the depthforge plugin there,
  3. writes depthforge_install.json pointing the plugin at this bundle's
     interpreter and app directory,
  4. downloads the OpenVINO models (unless already bundled/present),
  5. verifies that the whole chain actually imports and runs.

The bundle is used *in place*: the paths written into depthforge_install.json
point back here, so this folder must stay where it is after installation.

Usage (via the install.sh / install.bat wrappers):
    install.sh
    install.sh --gimp-dir /path/to/GIMP/3.2
    install.sh --skip-models
    uninstall.sh
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
PLUGIN_FOLDER_NAME = "depthforge"


# ── Console encoding ─────────────────────────────────────────────────────────
# On Windows our stdout is a file the Inno [Code] section reads back with
# LoadStringsFromFile, i.e. as ANSI, so Python encodes it in the locale code page
# (cp1250 on a Polish install). A character outside that page raises
# UnicodeEncodeError mid-print and kills the installer. Degrade to '?' instead --
# here and, via PYTHONIOENCODING, in download_models.py and every other child.
# Do not switch these streams to UTF-8: Inno would render the log as mojibake.
# Keep this module's own output ASCII-only regardless; this is the safety net for
# paths and tracebacks we do not control.
def _harden_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    # ":replace" overrides only the error handler, leaving the encoding alone.
    os.environ.setdefault("PYTHONIOENCODING", ":replace")


_harden_stdio()


# ── Pretty output ────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def section(msg: str) -> None:
    print(f"\n=== {msg} ===")


# ── Bundle layout ────────────────────────────────────────────────────────────

def read_manifest() -> dict:
    path = BUNDLE_ROOT / "bundle.json"
    if not path.is_file():
        fail(f"bundle.json missing at {path} - is this an unpacked DepthForge bundle?")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bundled_python() -> Path:
    """The interpreter running this script IS the bundled one."""
    return Path(sys.executable).resolve()


# ── GIMP discovery ───────────────────────────────────────────────────────────

def gimp_config_base() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "GIMP"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "GIMP"
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "GIMP"


def find_gimp_plugin_dir(explicit: str | None) -> tuple[Path, bool]:
    """Return (plug-ins dir, existed_already).

    GIMP 3.2 keeps its user config in <base>/3.2/. We prefer the highest
    version directory that actually exists, and fall back to 3.2 (creating it)
    when GIMP has never been launched.
    """
    if explicit:
        return Path(explicit).expanduser() / "plug-ins", True

    base = gimp_config_base()
    if base.is_dir():
        versions = []
        for d in base.iterdir():
            if d.is_dir() and d.name[:1].isdigit():
                try:
                    versions.append(([int(x) for x in d.name.split(".") if x.isdigit()], d))
                except ValueError:
                    continue
        if versions:
            versions.sort(reverse=True)
            return versions[0][1] / "plug-ins", True

    return base / "3.2" / "plug-ins", False


# ── Install steps ────────────────────────────────────────────────────────────

def install_plugin(plugin_dir: Path) -> Path:
    src = BUNDLE_ROOT / "plugin" / PLUGIN_FOLDER_NAME
    if not src.is_dir():
        fail(f"plugin source missing: {src}")
        sys.exit(1)

    dest = plugin_dir / PLUGIN_FOLDER_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        info(f"removing previous installation: {dest}")
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    ok(f"plugin copied -> {dest}")

    if platform.system() != "Windows":
        main_script = dest / "depthforge.py"
        if main_script.is_file():
            main_script.chmod(0o755)
            ok(f"chmod +x {main_script.name}")

    return dest


def write_install_json(dest_plugin_dir: Path, app_root: Path) -> None:
    """Point the plugin at this bundle: our interpreter, our app directory."""
    data = {
        "project_root": str(app_root),
        "venv_python": str(bundled_python()),
        "installed_by": "DepthForge bundle installer",
    }
    path = dest_plugin_dir / "depthforge_install.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    ok(f"depthforge_install.json -> {path}")
    info(f"project_root = {data['project_root']}")
    info(f"python       = {data['venv_python']}")


def models_present(app_root: Path) -> bool:
    required = [
        app_root / "models" / "dpt" / "openvino" / "dpt_large.bin",
        app_root / "models" / "midas" / "openvino" / "midas_v21_small_256.bin",
    ]
    return all(p.is_file() and p.stat().st_size > 1_000_000 for p in required)


def download_models(app_root: Path, release: str) -> bool:
    """Delegate to the project's own downloader, run by the bundled Python.

    download_models.py writes to paths relative to the CWD, hence cwd=app_root.
    """
    script = app_root / "download_models.py"
    if not script.is_file():
        fail(f"download_models.py missing at {script}")
        return False

    info("downloading OpenVINO models (~686 MB, one time)")
    r = subprocess.run(
        [str(bundled_python()), str(script), "--release", release],
        cwd=str(app_root),
    )
    return r.returncode == 0


def verify(app_root: Path) -> bool:
    """Import every runtime dependency and construct the real DepthForge object."""
    code = (
        "import sys, os\n"
        f"os.chdir(r'{app_root}')\n"
        f"sys.path.insert(0, r'{app_root / 'src'}')\n"
        "import numpy, cv2, openvino, scipy, stl\n"
        "from depth_forge import DepthForge\n"
        "from depth_pipeline import run_pipeline\n"
        "DepthForge()\n"
        "print('deps OK | numpy', numpy.__version__, '| cv2', cv2.__version__,\n"
        "      '| openvino', openvino.__version__)\n"
    )
    r = subprocess.run(
        [str(bundled_python()), "-c", code],
        capture_output=True, text=True, errors="replace", cwd=str(app_root),
    )
    if r.returncode != 0:
        fail("verification failed:")
        print(r.stderr.strip())
        return False
    for line in r.stdout.strip().splitlines():
        ok(line)
    return True


def uninstall(plugin_dir: Path) -> None:
    dest = plugin_dir / PLUGIN_FOLDER_NAME
    if dest.exists():
        shutil.rmtree(dest)
        ok(f"removed {dest}")
    else:
        info(f"nothing installed at {dest}")
    info("The bundle folder itself was left untouched - delete it manually to")
    info("reclaim the disk space.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="DepthForge bundle installer")
    ap.add_argument("--gimp-dir", default=None,
                    help=r"GIMP user config dir, e.g. %%APPDATA%%\GIMP\3.2 "
                         r"or ~/.config/GIMP/3.2")
    ap.add_argument("--skip-models", action="store_true",
                    help="do not download the models now")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    manifest = read_manifest()
    app_root = (BUNDLE_ROOT / manifest.get("app_root", "app")).resolve()

    print("=" * 64)
    print(f"  DepthForge {manifest.get('version', '?')} - {manifest.get('platform', '?')}")
    print(f"  bundled Python {manifest.get('python_version', '?')}")
    print("=" * 64)

    plugin_dir, existed = find_gimp_plugin_dir(args.gimp_dir)

    if args.uninstall:
        section("Uninstalling")
        uninstall(plugin_dir)
        return

    section("GIMP")
    info(f"plug-ins directory: {plugin_dir}")
    if not existed:
        warn("no existing GIMP config found - assuming GIMP 3.2.")
        warn("If GIMP is installed but uses another version, re-run with")
        warn("  --gimp-dir <path to your GIMP config dir>")

    section("Plugin")
    dest = install_plugin(plugin_dir)
    write_install_json(dest, app_root)

    section("Models")
    if models_present(app_root):
        ok("models already present in the bundle - nothing to download")
    elif args.skip_models:
        warn("skipped on request - the plugin will fall back to the synthetic")
        warn("estimator (flat, low-quality depth) until you run:")
        warn(f"  {bundled_python()} {app_root / 'download_models.py'}")
    else:
        if not download_models(app_root, manifest.get("model_release", "v0.1.0")):
            fail("model download failed - check your internet connection and re-run.")
            fail("The plugin is installed but will produce poor depth maps until")
            fail("the models are in place.")
            sys.exit(1)
        ok("models downloaded")

    section("Verification")
    if not verify(app_root):
        sys.exit(1)

    section("Done")
    print("  Restart GIMP, open an image, then:")
    print("    Filters > DepthForge > Generate Depth Map...")
    print()
    print(f"  IMPORTANT: do not move or delete this folder:")
    print(f"    {BUNDLE_ROOT}")
    print("  GIMP now points at the Python interpreter inside it.")
    print("  To move it, move the folder and re-run the installer from there.")


if __name__ == "__main__":
    main()
