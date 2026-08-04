#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – code-only update package
=====================================

Builds ``dist/DepthForge-<VERSION>-code.zip`` (plus its ``.sha256``): the slice
of a release that the GIMP plugin can install by itself, from inside GIMP.

Why this is a separate artifact
-------------------------------
A release changes ``src/`` and the plugin — a few hundred kilobytes. The private
CPython runtime, OpenVINO and the model weights that make up the rest of the
~1.2 GB offline bundle change almost never, and re-downloading them for a
pipeline fix is not something a museum machine on a metered link should have to
do. So the updater ships the code and leaves the runtime alone.

``requires_runtime`` is the valve that keeps that honest: the runtime is *not*
in this package, so a release whose code needs a new or newer dependency must
refuse to install this way and send the user to the full installer.
**Bump MIN_RUNTIME_VERSION whenever requirements.txt gains or bumps a
dependency** — that is the only thing standing between a code update and a
permanently broken install.

``config.json`` is deliberately *not* shipped
---------------------------------------------
It is the one file an end user may reasonably have edited (device, model paths),
and clobbering it would silently undo that. The consequence is a real constraint
on the pipeline: any config key added in a new version must be optional, with
its default in code, because an updated installation will still be running the
old ``config.json``. ``VERSION.json`` exists precisely so the version number can
advance without touching it.

Usage
-----
    python packaging/build_code_update.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bundle import (          # noqa: E402  – single source for VERSION
    DIST_DIR,
    PROJECT_ROOT,
    VERSION,
    human,
    log,
    section,
)

# Oldest installation whose runtime can execute this release's code. Compared
# against the installed VERSION.json / config.json version by the plugin's
# update helper, which refuses the update rather than half-applying it.
MIN_RUNTIME_VERSION = "1.7.0"

# Mapped onto the installation's project_root (the bundle's app/, or the repo
# root for a source checkout). Directories are replaced wholesale by the
# updater, so deletions upstream propagate.
UPDATE_DIRS = ["src"]
UPDATE_FILES = ["download_models.py"]

PLUGIN_FILE = Path("gimp_plugins") / "depthforge" / "depthforge.py"


def build() -> Path:
    name = f"DepthForge-{VERSION}-code"
    section(f"Building {name}.zip")

    staging = DIST_DIR / "staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    app = staging / "app"
    app.mkdir(parents=True)

    for rel in UPDATE_DIRS:
        shutil.copytree(
            PROJECT_ROOT / rel, app / rel,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        log(f"app/{rel}/")

    for rel in UPDATE_FILES:
        src = PROJECT_ROOT / rel
        if not src.is_file():
            raise SystemExit(f"missing {rel}")
        shutil.copy2(src, app / rel)
        log(f"app/{rel}")

    (app / "VERSION.json").write_text(
        json.dumps({"name": "DepthForge", "version": VERSION,
                    "source": "code-update"}, indent=2) + "\n",
        encoding="utf-8",
    )
    log("app/VERSION.json")

    plugin_src = PROJECT_ROOT / PLUGIN_FILE
    if not plugin_src.is_file():
        raise SystemExit(f"missing {PLUGIN_FILE}")
    plugin_dst = staging / "plugin" / "depthforge"
    plugin_dst.mkdir(parents=True)
    shutil.copy2(plugin_src, plugin_dst / "depthforge.py")
    log("plugin/depthforge/depthforge.py")

    (staging / "code.json").write_text(
        json.dumps({
            "name": "DepthForge",
            "kind": "code-update",
            "version": VERSION,
            "requires_runtime": MIN_RUNTIME_VERSION,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    log("code.json")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out = DIST_DIR / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))

    # Published alongside the zip and verified by the updater before anything is
    # overwritten. Without it the plugin refuses the update outright: the package
    # replaces src/ wholesale, which makes an unverified download remote code
    # execution regardless of HTTPS.
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (DIST_DIR / f"{out.name}.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )

    shutil.rmtree(staging, ignore_errors=True)

    log(f"{out.name}: {human(out.stat().st_size)}")
    log(f"sha256: {digest}")
    log(f"requires_runtime: {MIN_RUNTIME_VERSION}")
    return out


def main() -> None:
    out = build()
    section("Done")
    print(f"  {out}  ({human(out.stat().st_size)})")
    print(f"  {out}.sha256")


if __name__ == "__main__":
    main()
