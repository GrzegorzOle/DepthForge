#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – standalone bundle builder
======================================
Builds a self-contained DepthForge package for Linux and Windows.

The bundle ships its own CPython runtime (python-build-standalone) with every
runtime dependency pre-installed, so the end user needs **no Python at all** —
neither a system interpreter nor GIMP's bundled one, which is a different
version and cannot resolve our dependencies.

Both bundles are assembled *from Linux*: the Windows one is built by
downloading the win_amd64 wheels via ``pip --platform`` rather than by running
a Windows interpreter. Nothing here executes target-platform code, so a Windows
bundle can be produced on this machine but not smoke-tested on it.

Usage
-----
    python packaging/build_bundle.py                 # both platforms
    python packaging/build_bundle.py --target linux
    python packaging/build_bundle.py --target windows
    python packaging/build_bundle.py --with-models   # embed ~686 MB of models
    python packaging/build_bundle.py --no-archive    # leave staging dir only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_FILES = Path(__file__).resolve().parent / "bundle_files"
DIST_DIR = PROJECT_ROOT / "dist"

VERSION = "0.1.5"

# python-build-standalone: relocatable CPython, identical layout story on both
# platforms. Pinned on purpose — "latest" would make builds unreproducible.
PBS_RELEASE = "20260623"
PYTHON_VERSION = "3.12.13"

TARGETS = {
    "linux": {
        "pbs_triple": "x86_64-unknown-linux-gnu",
        "archive": "tar.gz",
        "python_exe": "python/bin/python3",
        "site_packages": "python/lib/python3.12/site-packages",
        # openvino ships manylinux_2_28 only (glibc >= 2.28); the others are
        # for packages that still target the older baseline.
        "pip_platforms": [
            "manylinux_2_28_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ],
        "label": "linux-x86_64",
    },
    "windows": {
        "pbs_triple": "x86_64-pc-windows-msvc",
        "archive": "zip",
        "python_exe": "python/python.exe",
        "site_packages": "python/Lib/site-packages",
        "pip_platforms": ["win_amd64"],
        "label": "windows-x86_64",
    },
}

# Project files that make up the runnable application inside the bundle.
APP_FILES = ["config.json", "download_models.py", "LICENSE"]
APP_DIRS = ["src"]

MODEL_FILES = [
    "models/dpt/openvino/dpt_large.bin",
    "models/dpt/openvino/dpt_large.xml",
    "models/midas/openvino/midas_v21_small_256.bin",
    "models/midas/openvino/midas_v21_small_256.xml",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def section(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download(url: str, dest: Path) -> Path:
    """Download url to dest, reusing an existing complete file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"cached: {dest.name} ({human(dest.stat().st_size)})")
        return dest

    log(f"downloading: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(block: int, size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * size, total)
        pct = done / total * 100
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        print(f"\r  [{bar}] {pct:5.1f}%  {human(done)}", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)
    log(f"got {dest.name} ({human(dest.stat().st_size)})")
    return dest


def fetch_runtime(triple: str, cache: Path) -> Path:
    """Download and cache the python-build-standalone tarball for a triple.

    The 'stripped' variant matters: the default install_only build ships
    unstripped debug symbols (libpython alone is 208 MB), which no end user
    needs and which triples the bundle size.
    """
    name = f"cpython-{PYTHON_VERSION}+{PBS_RELEASE}-{triple}-install_only_stripped.tar.gz"
    url = (
        f"https://github.com/astral-sh/python-build-standalone/releases/"
        f"download/{PBS_RELEASE}/{name}"
    )
    return download(url, cache / name)


def extract_runtime(tarball: Path, staging: Path) -> None:
    """Extract the runtime; the tarball contains a top-level 'python/' dir."""
    log(f"extracting runtime → {staging / 'python'}")
    with tarfile.open(tarball) as tf:
        # filter='data' refuses absolute paths / traversal; the PBS archives are
        # plain files so nothing legitimate is dropped.
        tf.extractall(staging, filter="data")


def pip_install_deps(target: dict, staging: Path) -> None:
    """Cross-install the wheels for the target platform into the bundled runtime.

    Uses the *host* pip with --platform/--target: we cannot execute the target
    interpreter here, so resolution is done by tag, not by running it.
    """
    site = staging / target["site_packages"]
    site.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(site),
        "--only-binary=:all:",
        "--implementation", "cp",
        "--python-version", "3.12",
        "--upgrade",
        "-r", str(PROJECT_ROOT / "requirements.txt"),
    ]
    for plat in target["pip_platforms"]:
        cmd += ["--platform", plat]

    log("pip install " + " ".join(target["pip_platforms"]))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"pip install failed for {target['label']}")

    log(f"site-packages: {human(dir_size(site))}")


def copy_app(staging: Path, with_models: bool) -> None:
    """Copy the DepthForge application itself into the bundle."""
    app = staging / "app"
    app.mkdir(parents=True, exist_ok=True)

    for name in APP_DIRS:
        shutil.copytree(
            PROJECT_ROOT / name, app / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        log(f"app/{name}/")

    for name in APP_FILES:
        src = PROJECT_ROOT / name
        if src.is_file():
            shutil.copy2(src, app / name)
            log(f"app/{name}")

    if with_models:
        for rel in MODEL_FILES:
            src = PROJECT_ROOT / rel
            if not src.is_file():
                raise SystemExit(
                    f"--with-models given but {rel} is missing. "
                    f"Run: python download_models.py"
                )
            dest = app / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            log(f"app/{rel} ({human(src.stat().st_size)})")


def copy_plugin_and_installer(staging: Path, target: dict) -> None:
    """Copy the GIMP plugin folder, the installer and the user docs."""
    shutil.copytree(
        PROJECT_ROOT / "gimp_plugins" / "depthforge",
        staging / "plugin" / "depthforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "depthforge_install.json"),
    )
    log("plugin/depthforge/")

    shutil.copy2(BUNDLE_FILES / "bundle_install.py", staging / "bundle_install.py")

    for doc in ("INSTALL_PL.md", "INSTALL_EN.md"):
        shutil.copy2(BUNDLE_FILES / doc, staging / doc)
        log(doc)

    if target["archive"] == "zip":
        shutil.copy2(BUNDLE_FILES / "install.bat", staging / "install.bat")
        shutil.copy2(BUNDLE_FILES / "uninstall.bat", staging / "uninstall.bat")
        log("install.bat / uninstall.bat")
    else:
        for name in ("install.sh", "uninstall.sh"):
            dest = staging / name
            shutil.copy2(BUNDLE_FILES / name, dest)
            dest.chmod(0o755)
        log("install.sh / uninstall.sh")


def write_manifest(staging: Path, target: dict, with_models: bool) -> None:
    manifest = {
        "name": "DepthForge",
        "version": VERSION,
        "platform": target["label"],
        "python_version": PYTHON_VERSION,
        "python_exe": target["python_exe"],
        "app_root": "app",
        "models_included": with_models,
        "model_release": "v0.1.0",
    }
    (staging / "bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log("bundle.json")


def prune(staging: Path) -> None:
    """Drop weight that no end user needs: test suites, headers, caches."""
    before = dir_size(staging)
    patterns = ["__pycache__", "*.pyc", "*.pyo"]
    for pat in patterns:
        for p in staging.rglob(pat):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)

    # pip/setuptools inside the runtime are dead weight — the bundle never
    # installs anything at runtime.
    for site in staging.rglob("site-packages"):
        for junk in ("pip", "pip-*.dist-info", "setuptools", "setuptools-*.dist-info",
                     "pkg_resources", "wheel", "wheel-*.dist-info"):
            for p in site.glob(junk):
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()

    # numpy/scipy ship test suites that rival the code in size.
    for name in ("numpy/tests", "numpy/_core/tests", "scipy/**/tests"):
        for p in staging.rglob(name):
            shutil.rmtree(p, ignore_errors=True)

    prune_openvino(staging)

    after = dir_size(staging)
    log(f"pruned {human(before - after)} → {human(after)}")


def prune_openvino(staging: Path) -> None:
    """Drop OpenVINO device plugins and model frontends the pipeline never uses.

    depth_forge.py only ever calls compile_model(<IR .xml>, device) with
    device='CPU' from config.json, so the CPU plugin and the IR frontend are
    the only ones on the code path. The GPU plugin alone is 44 MB.

    Consequence to keep in mind: setting model.depth_estimation.device to "GPU"
    in a bundled config.json will not work — that is intentional, the bundle is
    a CPU-only artifact. The build verifies this by running the real pipeline.
    """
    dropped = 0
    for libs in staging.rglob("openvino/libs"):
        for pat in (
            "*intel_gpu_plugin*", "*intel_npu_plugin*",
            "*tensorflow_frontend*", "*tensorflow_lite_frontend*",
            "*pytorch_frontend*", "*paddle_frontend*", "*jax_frontend*",
        ):
            for p in libs.glob(pat):
                dropped += p.stat().st_size
                p.unlink()
    if dropped:
        log(f"openvino: dropped {human(dropped)} of unused plugins/frontends")


def make_archive(staging: Path, target: dict, name: str) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if target["archive"] == "zip":
        out = DIST_DIR / f"{name}.zip"
        log(f"zipping → {out.name} (this takes a minute)")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(staging.rglob("*")):
                if f.is_file() or f.is_symlink():
                    zf.write(f, Path(name) / f.relative_to(staging))
    else:
        out = DIST_DIR / f"{name}.tar.gz"
        log(f"tarring → {out.name} (this takes a minute)")
        with tarfile.open(out, "w:gz", compresslevel=6) as tf:
            tf.add(staging, arcname=name)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (DIST_DIR / f"{out.name}.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    log(f"{out.name}: {human(out.stat().st_size)}")
    log(f"sha256: {digest}")
    return out


def verify_linux_bundle(staging: Path, with_models: bool) -> None:
    """Exercise the bundled Linux runtime for real: imports, then a full run.

    Only possible for the Linux target — we are on Linux. The Windows bundle
    ships unverified by construction.

    Importing the deps is not enough: it would not touch OpenVINO inference and
    so would not catch a bad plugin prune. So we run the actual tactile pipeline
    on a sample image, borrowing the project's models when the bundle does not
    carry its own.
    """
    py = staging / "python" / "bin" / "python3"
    app = staging / "app"

    code = (
        "import sys, numpy, cv2, openvino, scipy, stl;"
        "print('python', sys.version.split()[0]);"
        "print('numpy', numpy.__version__);"
        "print('cv2', cv2.__version__);"
        "print('openvino', openvino.__version__);"
        "print('scipy', scipy.__version__)"
    )
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"bundled runtime failed to import deps:\n{r.stderr}")
    for line in r.stdout.strip().splitlines():
        log(f"verified: {line}")

    borrowed = False
    models_dir = app / "models"
    if not with_models and not models_dir.exists() and (PROJECT_ROOT / "models").is_dir():
        models_dir.symlink_to(PROJECT_ROOT / "models")
        borrowed = True

    if not (models_dir / "dpt" / "openvino" / "dpt_large.xml").is_file():
        log("WARN: no models available — skipping the end-to-end pipeline check")
        return

    sample = PROJECT_ROOT / "data" / "museum_sample_input.jpg"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = (
                "import sys, os\n"
                f"sys.path.insert(0, r'{app / 'src'}')\n"
                "from depth_pipeline import run_pipeline_tactile\n"
                f"run_pipeline_tactile(r'{sample}', r'{tmp}', width_mm=200, relief_mm=7)\n"
                f"print('artifacts:', sorted(os.listdir(r'{tmp}')))\n"
            )
            log("running the real tactile pipeline through the bundled runtime…")
            r = subprocess.run([str(py), "-c", run], capture_output=True, text=True,
                               cwd=str(app), timeout=900)
            if r.returncode != 0:
                raise SystemExit(
                    "bundled runtime failed the end-to-end pipeline run:\n"
                    + r.stderr[-3000:]
                )

            # An STL alone proves nothing: with the models missing the pipeline
            # silently drops to the synthetic estimator and still exits 0. The
            # midas/dpt tasks are only scheduled when their OpenVINO models
            # compile, so their labels are the real proof.
            expected = ("OpenVINO DPT Large", "OpenVINO MiDaS v2.1 Small")
            missing = [lbl for lbl in expected if f"✓ {lbl}" not in r.stdout]
            if missing and (with_models or borrowed):
                raise SystemExit(
                    f"models were available but {', '.join(missing)} never ran — "
                    "the bundle fell back to the synthetic estimator.\n\n"
                    + r.stdout[-2500:]
                )
            if not missing:
                log("verified: OpenVINO DPT Large + MiDaS both ran (real weights)")

            stl = [f for f in os.listdir(tmp) if f.endswith(".stl")]
            if not stl:
                raise SystemExit(f"pipeline produced no STL:\n{r.stdout[-2000:]}")
            log(f"verified: end-to-end run produced {stl}")
    finally:
        if borrowed:
            models_dir.unlink()


# ── Build ────────────────────────────────────────────────────────────────────

def build(target_name: str, with_models: bool, archive: bool, cache: Path) -> None:
    target = TARGETS[target_name]
    suffix = "-offline" if with_models else ""
    name = f"DepthForge-{VERSION}-{target['label']}{suffix}"

    section(f"Building {name}")

    staging = DIST_DIR / "staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    tarball = fetch_runtime(target["pbs_triple"], cache)
    extract_runtime(tarball, staging)
    pip_install_deps(target, staging)
    copy_app(staging, with_models)
    copy_plugin_and_installer(staging, target)
    write_manifest(staging, target, with_models)
    prune(staging)

    if target_name == "linux":
        verify_linux_bundle(staging, with_models)
    else:
        log("NOTE: Windows bundle cannot be executed here — untested by construction")

    log(f"staging total: {human(dir_size(staging))}")

    if archive:
        make_archive(staging, target, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build standalone DepthForge bundles")
    ap.add_argument("--target", choices=["linux", "windows", "both"], default="both")
    ap.add_argument("--with-models", action="store_true",
                    help="embed the OpenVINO models (~686 MB) for offline install")
    ap.add_argument("--no-archive", action="store_true",
                    help="leave the staging directory, skip tar/zip")
    ap.add_argument("--cache", default=str(DIST_DIR / "cache"),
                    help="where to cache downloaded runtimes")
    args = ap.parse_args()

    cache = Path(args.cache)
    targets = ["linux", "windows"] if args.target == "both" else [args.target]

    for t in targets:
        build(t, args.with_models, not args.no_archive, cache)

    section("Done")
    if not args.no_archive:
        for f in sorted(DIST_DIR.glob("DepthForge-*")):
            if f.is_file() and not f.name.endswith(".sha256"):
                print(f"  {f}  ({human(f.stat().st_size)})")


if __name__ == "__main__":
    main()
