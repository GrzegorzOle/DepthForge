#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – AppImage builder
=============================
Packs the offline Linux bundle into a single self-contained .AppImage: private
CPython, every dependency, the OpenVINO models, the GIMP plugin and the installer.

The AppImage is not a launcher — it is a **Python interpreter in a single file**.
AppRun forwards argv to the bundled interpreter, so the GIMP plugin's
``subprocess.run([venv_python, helper.py, ...])`` works with the .AppImage path as
venv_python and no plugin changes at all. See packaging/appimage_files/AppRun for
why the state directory it maintains is load-bearing.

Prerequisites
-------------
* The Linux offline staging tree, i.e. run first:
      python packaging/build_bundle.py --target linux --with-models --no-archive
* appimagetool — fetched automatically into the cache if absent.
* FUSE to *run* the result (to build it we pass --appimage-extract-and-run).

Usage
-----
    python packaging/build_appimage.py
    python packaging/build_appimage.py --no-verify     # skip the end-to-end run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPIMAGE_FILES = Path(__file__).resolve().parent / "appimage_files"
ASSETS = Path(__file__).resolve().parent / "assets"
DIST_DIR = PROJECT_ROOT / "dist"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bundle import VERSION, human, dir_size, log, section, download  # noqa: E402

APPIMAGETOOL_URL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/"
    "appimagetool-x86_64.AppImage"
)

DESKTOP = """[Desktop Entry]
Type=Application
Name=DepthForge
GenericName=Tactile depth map generator
Comment=Generate depth maps and printable tactile reliefs from flat images
Exec=AppRun
Icon=depthforge
Categories=Graphics;3DGraphics;Photography;
Terminal=true
Keywords=depth;relief;tactile;3D;STL;GIMP;museum;accessibility;
"""


def fetch_appimagetool(cache: Path) -> Path:
    tool = download(APPIMAGETOOL_URL, cache / "appimagetool-x86_64.AppImage")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


def build_appdir(staging: Path, appdir: Path) -> None:
    """Assemble the AppDir from the already-built offline staging tree."""
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)

    # The heavy payload: reflink/hardlink where possible so we do not copy a GB.
    for name in ("python", "app", "plugin"):
        src = staging / name
        if not src.is_dir():
            raise SystemExit(
                f"{src} missing — run:\n"
                f"  python packaging/build_bundle.py --target linux "
                f"--with-models --no-archive"
            )
        log(f"AppDir/{name}/  ({human(dir_size(src))})")
        subprocess.run(["cp", "-a", "--reflink=auto", str(src), str(appdir / name)],
                       check=True)

    shutil.copy2(staging / "bundle.json", appdir / "bundle.json")
    shutil.copy2(APPIMAGE_FILES / "appimage_install.py", appdir / "appimage_install.py")
    for doc in ("INSTALL_PL.md", "INSTALL_EN.md"):
        if (staging / doc).is_file():
            shutil.copy2(staging / doc, appdir / doc)

    apprun = appdir / "AppRun"
    shutil.copy2(APPIMAGE_FILES / "AppRun", apprun)
    apprun.chmod(0o755)
    log("AppDir/AppRun (argv passthrough → bundled python)")

    (appdir / "depthforge.desktop").write_text(DESKTOP, encoding="utf-8")
    shutil.copy2(ASSETS / "depthforge.png", appdir / "depthforge.png")
    # appimagetool also wants the icon under the standard hicolor path.
    icon_dir = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    icon_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "depthforge.png", icon_dir / "depthforge.png")
    log("AppDir/depthforge.desktop + icon")

    # DepthForge() mkdir()s every directory named in config.json on construction.
    # Inside the read-only squashfs that would raise, so make sure they already
    # exist — the installer's state dir covers the writable ones, but the CLI
    # (--df-cli) runs against app/ directly.
    for name in ("data", "output"):
        (appdir / "app" / name).mkdir(parents=True, exist_ok=True)


def make_appimage(appdir: Path, tool: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    env = dict(os.environ)
    # We may be inside a container/CI without FUSE; this makes appimagetool
    # unpack itself instead of mounting. It does not affect the artifact.
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    env["ARCH"] = "x86_64"

    log(f"appimagetool → {out.name} (squashfs compression, this takes a while)")
    r = subprocess.run(
        [str(tool), "--comp", "zstd", "--no-appstream", str(appdir), str(out)],
        env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(
            "appimagetool failed:\n" + (r.stderr or r.stdout)[-3000:]
        )
    out.chmod(0o755)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / f"{out.name}.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    log(f"{out.name}: {human(out.stat().st_size)}")
    log(f"sha256: {digest}")
    return out


def verify_appimage(appimage: Path) -> None:
    """Prove the finished artifact behaves as an interpreter *and* renders depth.

    Two things are checked, and the second is the one that matters. A bad prune or
    a missing model does not crash the pipeline — DepthForge silently degrades to
    its synthetic estimator and still exits 0 with a plausible-looking map, so
    "an STL came out" proves nothing.

    The load-bearing signal is the per-backend label: depth_pipeline only appends
    the midas/dpt tasks if the OpenVINO models actually compiled, so seeing both
    labels printed is proof the real weights were found inside the image.
    """
    section("Verifying the AppImage")

    r = subprocess.run([str(appimage), "-c",
                        "import sys, numpy, cv2, openvino, scipy, stl;"
                        "print('python', sys.version.split()[0]);"
                        "print('openvino', openvino.__version__)"],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"AppImage is not usable as an interpreter:\n{r.stderr[-2000:]}")
    for line in r.stdout.strip().splitlines():
        log(f"verified: {line}")

    sample = PROJECT_ROOT / "data" / "museum_sample_input.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "import sys, os\n"
            "sys.path.insert(0, os.path.join(os.environ['APPDIR'], 'app', 'src'))\n"
            "os.chdir(os.path.join(os.environ['APPDIR'], 'app'))\n"
            "from depth_pipeline import run_pipeline_tactile\n"
            f"run_pipeline_tactile(r'{sample}', r'{tmp}', width_mm=200, relief_mm=7)\n"
            f"print('artifacts:', sorted(os.listdir(r'{tmp}')))\n"
        )
        log("running the real tactile pipeline through the AppImage…")
        r = subprocess.run([str(appimage), "-c", code],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise SystemExit(
                "AppImage failed the end-to-end pipeline run:\n" + r.stderr[-3000:]
            )

        missing = [label for label in ("OpenVINO DPT Large",
                                       "OpenVINO MiDaS v2.1 Small")
                   if f"✓ {label}" not in r.stdout]
        if missing:
            raise SystemExit(
                f"the pipeline ran but never loaded {', '.join(missing)} — the models "
                "inside the AppImage are not being found, and it silently fell back "
                "to the synthetic estimator.\n\n" + r.stdout[-2500:]
            )
        log("verified: OpenVINO DPT Large + MiDaS both ran (real weights, not synthetic)")

        stls = sorted(f for f in os.listdir(tmp) if f.endswith(".stl"))
        if not stls:
            raise SystemExit(f"pipeline produced no STL:\n{r.stdout[-2000:]}")
        log(f"verified: end-to-end run produced {stls}")


def verify_install_flow(appimage: Path) -> None:
    """Install into a throwaway HOME and confirm the plugin wiring it writes.

    This is what catches the AppImage-specific trap: venv_python must be the
    .AppImage itself and project_root must resolve through the state dir, not into
    this run's mountpoint.
    """
    section("Verifying the install flow")

    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ)
        env["HOME"] = home
        env["XDG_DATA_HOME"] = str(Path(home) / ".local" / "share")
        env["XDG_CONFIG_HOME"] = str(Path(home) / ".config")

        r = subprocess.run([str(appimage), "--df-install"],
                           capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0:
            raise SystemExit(f"--df-install failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")

        import json
        cfg = (Path(env["XDG_CONFIG_HOME"]) / "GIMP" / "3.2" / "plug-ins"
               / "depthforge" / "depthforge_install.json")
        if not cfg.is_file():
            raise SystemExit(f"installer wrote no depthforge_install.json at {cfg}")
        data = json.loads(cfg.read_text())

        if Path(data["venv_python"]).resolve() != appimage.resolve():
            raise SystemExit(
                f"venv_python should be the AppImage itself, got {data['venv_python']}"
            )
        log(f"verified: venv_python = {data['venv_python']}")

        root = Path(data["project_root"])
        if not (root / "config.json").is_file():
            raise SystemExit(f"project_root has no config.json: {root}")
        if "/.mount_" in str(root.resolve()):
            raise SystemExit(f"project_root points into a transient mount: {root}")
        log(f"verified: project_root = {root} (stable, config.json present)")

        # The symlinks must resolve into a mount that no longer exists *now* — the
        # point is that AppRun re-aims them, so check they resolve when going
        # through the AppImage rather than from here.
        probe = (
            "import os\n"
            "p = os.path.join(os.environ['DF_ROOT'], 'models', 'dpt', 'openvino',"
            " 'dpt_large.xml')\n"
            "print('MODELS_RESOLVE:', os.path.isfile(p))\n"
        )
        env["DF_ROOT"] = str(root)
        r = subprocess.run([str(appimage), "-c", probe],
                           capture_output=True, text=True, timeout=300, env=env)
        if "MODELS_RESOLVE: True" not in r.stdout:
            raise SystemExit(
                "the state dir's model symlink does not resolve on a later run — "
                "AppRun is not re-aiming _appdir.\n" + r.stdout + r.stderr[-1500:]
            )
        log("verified: state-dir symlinks re-resolve on a fresh mount")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the DepthForge AppImage")
    ap.add_argument("--staging", default=None,
                    help="offline linux staging dir (default: the 0.x.y one in dist/)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--cache", default=str(DIST_DIR / "cache"))
    args = ap.parse_args()

    staging = Path(args.staging) if args.staging else (
        DIST_DIR / "staging" / f"DepthForge-{VERSION}-linux-x86_64-offline"
    )
    out = DIST_DIR / f"DepthForge-{VERSION}-x86_64.AppImage"

    section(f"Building {out.name}")
    log(f"staging: {staging}")

    tool = fetch_appimagetool(Path(args.cache))

    appdir = DIST_DIR / "staging" / "AppDir"
    build_appdir(staging, appdir)
    log(f"AppDir total: {human(dir_size(appdir))}")

    make_appimage(appdir, tool, out)

    if not args.no_verify:
        verify_appimage(out)
        verify_install_flow(out)

    section("Done")
    print(f"  {out}  ({human(out.stat().st_size)})")


if __name__ == "__main__":
    main()
