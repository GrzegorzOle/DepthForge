#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – AppImage installer
===============================
Runs under the CPython interpreter inside the AppImage (AppRun execs us).

This is the AppImage counterpart of bundle_files/bundle_install.py. It differs in
the two things that make an AppImage an AppImage:

  * ``venv_python`` is the **.AppImage file itself**, not the interpreter inside
    it. AppRun forwards argv to the bundled interpreter, so the plugin's
    ``subprocess.run([venv_python, helper.py, ...])`` works unchanged, and the
    path stays valid across runs. ``sys.executable`` would point into this run's
    throwaway mountpoint.

  * ``project_root`` cannot live inside the AppImage either. The plugin resolves
    it (and stat()s config.json there) before it ever execs us, so it must be a
    path that survives between runs — and our mount is a fresh
    /tmp/.mount_XXXXXX each time. So we build a small **state directory**:

        ~/.local/share/DepthForge/
            _appdir -> /tmp/.mount_XXXXXX      (re-aimed by AppRun every run)
            app/
                config.json                    (real copy, so the plugin's check passes)
                src     -> ../_appdir/app/src
                models  -> ../_appdir/app/models
                data/                          (real, writable)
                output/                        (real, writable)

    The relative symlinks resolve through _appdir, so they always reach the
    current mount. Keeping data/ and output/ real also matters: DepthForge()
    mkdir()s every directory named in config.json on construction, which would
    fail on the read-only squashfs.

Usage:
    ./DepthForge-x.y.z-x86_64.AppImage                 # install
    ./DepthForge-x.y.z-x86_64.AppImage --df-install --gimp-dir ~/.config/GIMP/3.2
    ./DepthForge-x.y.z-x86_64.AppImage --df-uninstall
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

APPDIR = Path(os.environ.get("APPDIR", Path(__file__).resolve().parent))
PLUGIN_FOLDER_NAME = "depthforge"

# Directories from config.json that must be real and writable rather than
# symlinks into the read-only image.
WRITABLE_DIRS = ("data", "output")
# Directories that live in the image and are reached through _appdir.
LINKED_DIRS = ("src", "models")


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


# ── Locations ────────────────────────────────────────────────────────────────

def state_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(base) / "DepthForge"


def appimage_path() -> Path:
    """The .AppImage file the user launched.

    $APPIMAGE is set by the AppImage runtime. Without it we are probably running
    from an extracted AppDir (APPIMAGE_EXTRACT_AND_RUN, or a build-time test),
    where the interpreter path is stable enough to use directly.
    """
    env = os.environ.get("APPIMAGE", "").strip()
    if env:
        return Path(env).resolve()
    warn("$APPIMAGE unset — assuming an extracted AppDir, not a real AppImage")
    return (APPDIR / "AppRun").resolve()


def read_manifest() -> dict:
    path = APPDIR / "bundle.json"
    if not path.is_file():
        fail(f"bundle.json missing at {path} — broken AppImage?")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── GIMP discovery (mirrors bundle_install.py) ───────────────────────────────

def find_gimp_plugin_dir(explicit: str | None) -> tuple[Path, bool]:
    if explicit:
        return Path(explicit).expanduser() / "plug-ins", True

    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    base = Path(xdg) / "GIMP"
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


# ── State directory ──────────────────────────────────────────────────────────

def build_state_dir() -> Path:
    """Create the stable app root the plugin will chdir() into. Returns its path."""
    state = state_dir()
    app = state / "app"
    app.mkdir(parents=True, exist_ok=True)

    # Aim the pointer at this run's mount, so the relative symlinks below resolve
    # immediately (AppRun re-aims it on every later run).
    link = state / "_appdir"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(APPDIR)
    ok(f"_appdir → {APPDIR}")

    # A real config.json: the plugin checks for this file to accept project_root,
    # and it does so before AppRun can refresh anything.
    shutil.copy2(APPDIR / "app" / "config.json", app / "config.json")
    ok("app/config.json (real copy)")

    for name in LINKED_DIRS:
        dest = app / name
        if dest.is_symlink() or dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        # Relative, so it resolves through _appdir to whatever is mounted now.
        dest.symlink_to(Path("..") / "_appdir" / "app" / name)
        ok(f"app/{name} → ../_appdir/app/{name}")

    for name in WRITABLE_DIRS:
        (app / name).mkdir(parents=True, exist_ok=True)
        ok(f"app/{name}/ (writable)")

    return app


def install_plugin(plugin_dir: Path) -> Path:
    src = APPDIR / "plugin" / PLUGIN_FOLDER_NAME
    if not src.is_dir():
        fail(f"plugin source missing: {src}")
        sys.exit(1)

    dest = plugin_dir / PLUGIN_FOLDER_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        info(f"removing previous installation: {dest}")
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    (dest / "depthforge.py").chmod(0o755)
    ok(f"plugin copied → {dest}")
    return dest


def write_install_json(dest_plugin_dir: Path, app_root: Path) -> None:
    data = {
        "project_root": str(app_root),
        "venv_python": str(appimage_path()),
        "installed_by": "DepthForge AppImage installer",
    }
    path = dest_plugin_dir / "depthforge_install.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    ok(f"depthforge_install.json → {path}")
    info(f"project_root = {data['project_root']}")
    info(f"python       = {data['venv_python']}")


def verify(app_root: Path) -> bool:
    """Run the real chain the way the plugin will: through the AppImage itself.

    Going through appimage_path() rather than sys.executable is the whole point —
    it proves the argv passthrough and the symlink indirection work together.
    Importing the deps alone would prove neither.
    """
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
        [str(appimage_path()), "-c", code],
        capture_output=True, text=True, cwd=str(app_root),
    )
    if r.returncode != 0:
        fail("verification failed:")
        print(r.stderr.strip())
        return False
    for line in r.stdout.strip().splitlines():
        ok(line)
    return True


def uninstall(plugin_dir: Path, keep_state: bool) -> None:
    dest = plugin_dir / PLUGIN_FOLDER_NAME
    if dest.exists():
        shutil.rmtree(dest)
        ok(f"removed {dest}")
    else:
        info(f"nothing installed at {dest}")

    state = state_dir()
    if keep_state:
        info(f"kept {state} (--keep-state)")
    elif state.exists():
        shutil.rmtree(state)
        ok(f"removed {state}")

    info("The .AppImage file itself was left alone — delete it to reclaim the space.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="DepthForge AppImage installer")
    ap.add_argument("--gimp-dir", default=None,
                    help="GIMP user config dir, e.g. ~/.config/GIMP/3.2")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--keep-state", action="store_true",
                    help="on uninstall, keep ~/.local/share/DepthForge")
    args = ap.parse_args()

    manifest = read_manifest()
    print("=" * 64)
    print(f"  DepthForge {manifest.get('version', '?')} – AppImage")
    print(f"  bundled Python {manifest.get('python_version', '?')}")
    print("=" * 64)

    plugin_dir, existed = find_gimp_plugin_dir(args.gimp_dir)

    if args.uninstall:
        section("Uninstalling")
        uninstall(plugin_dir, args.keep_state)
        return

    section("GIMP")
    info(f"plug-ins directory: {plugin_dir}")
    if not existed:
        warn("no existing GIMP config found — assuming GIMP 3.2.")
        warn("If GIMP is installed but uses another version, re-run with")
        warn("  --df-install --gimp-dir <path to your GIMP config dir>")

    section("State directory")
    app_root = build_state_dir()

    section("Plugin")
    dest = install_plugin(plugin_dir)
    write_install_json(dest, app_root)

    section("Models")
    dpt = APPDIR / "app" / "models" / "dpt" / "openvino" / "dpt_large.bin"
    if dpt.is_file() and dpt.stat().st_size > 1_000_000:
        ok("models are inside the AppImage — nothing to download")
    else:
        fail("models missing from the AppImage — this build is not the offline one")
        sys.exit(1)

    section("Verification")
    if not verify(app_root):
        sys.exit(1)

    section("Done")
    print("  Restart GIMP, open an image, then:")
    print("    Filters → DepthForge → Generate Depth Map…")
    print()
    print("  IMPORTANT: do not move or rename this file:")
    print(f"    {appimage_path()}")
    print("  GIMP now calls it as its Python interpreter. If you move it,")
    print("  just run it again from the new location.")


if __name__ == "__main__":
    main()
