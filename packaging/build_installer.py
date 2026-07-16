#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – Windows installer builder
======================================
Compiles packaging/innosetup/depthforge.iss into a single setup .exe wrapping the
offline Windows bundle.

Inno Setup is a Windows tool, so it is run through the amake/innosetup container
(Inno Setup 6 under wine) rather than by installing wine system-wide. The
container mounts the project at /work and iscc is invoked with paths relative to
it.

This means the .exe is **cross-built and unverifiable here**, the same caveat the
Windows bundle already carries: nothing on this machine can execute it. What is
checked below is only what can be checked from Linux — that the compiler
succeeded, and that the payload it was handed is the offline one.

Prerequisites
-------------
* podman (or docker)
* The Windows offline staging tree:
      python packaging/build_bundle.py --target windows --with-models --no-archive

Usage
-----
    python packaging/build_installer.py
    python packaging/build_installer.py --engine docker
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
ISS = "packaging/innosetup/depthforge.iss"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bundle import VERSION, human, dir_size, log, section  # noqa: E402

IMAGE = "docker.io/amake/innosetup:latest"

# The container mounts the project at /work, which wine exposes as Z:\work. The
# -D values below must be Windows-style: iscc treats a path without a drive
# letter as relative to the .iss file, so a bare "/work/dist" silently becomes
# packaging\innosetup\work\dist and the build fails somewhere far from the cause.
WIN_WORK = r"Z:\work"


def pick_engine(preferred: str | None) -> str:
    candidates = [preferred] if preferred else ["podman", "docker"]
    for c in candidates:
        if c and shutil.which(c):
            return c
    raise SystemExit(
        "neither podman nor docker found — one of them is needed to run "
        "Inno Setup under wine"
    )


def check_payload(staging: Path) -> None:
    """Refuse to build an 'offline' installer around a bundle without models."""
    if not staging.is_dir():
        raise SystemExit(
            f"{staging} missing — run:\n"
            f"  python packaging/build_bundle.py --target windows "
            f"--with-models --no-archive"
        )

    required = [
        staging / "python" / "python.exe",
        staging / "bundle_install.py",
        staging / "app" / "config.json",
        staging / "plugin" / "depthforge" / "depthforge.py",
        staging / "app" / "models" / "dpt" / "openvino" / "dpt_large.bin",
        staging / "app" / "models" / "midas" / "openvino" / "midas_v21_small_256.bin",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise SystemExit(
            "payload is incomplete — missing:\n"
            + "\n".join(f"  {p.relative_to(staging)}" for p in missing)
        )

    dpt = staging / "app" / "models" / "dpt" / "openvino" / "dpt_large.bin"
    if dpt.stat().st_size < 100_000_000:
        raise SystemExit(f"{dpt.name} is only {human(dpt.stat().st_size)} — truncated?")

    log(f"payload: {staging.name} ({human(dir_size(staging))}, models embedded)")


def compile_iss(engine: str, staging: Path, out_base: str) -> Path:
    win_src = WIN_WORK + "\\" + str(staging.relative_to(PROJECT_ROOT)).replace("/", "\\")
    cmd = [engine, "run", "--rm"]

    if engine == "podman":
        # The image runs iscc as its own uid 1000. Under rootless podman our uid
        # maps to container root, so the bind mount arrives owned by "root" and
        # that uid 1000 cannot write the .exe into dist/ — the failure surfaces
        # late and unhelpfully as "Error 5: Access denied". keep-id lines the two
        # uids up. Docker (root daemon) does not need this.
        cmd += ["--userns=keep-id"]

    cmd += [
        "-v", f"{PROJECT_ROOT}:/work:z",
        "-w", "/work",
        IMAGE,
        ISS,
        f"/DAppVersion={VERSION}",
        f"/DSourceDir={win_src}",
        f"/DOutputDir={WIN_WORK}\\dist",
        f"/DOutputBaseName={out_base}",
    ]

    log(f"{engine} run {IMAGE}")
    log("iscc — compressing ~1.2 GB, expect several minutes")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            "iscc failed:\n" + (r.stdout or "")[-2000:] + "\n" + (r.stderr or "")[-3000:]
        )

    for line in (r.stdout or "").strip().splitlines()[-6:]:
        log(line.strip())

    out = DIST_DIR / f"{out_base}.exe"
    if not out.is_file():
        raise SystemExit(f"iscc reported success but {out} does not exist")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the DepthForge Windows installer")
    ap.add_argument("--engine", choices=["podman", "docker"], default=None)
    ap.add_argument("--staging", default=None)
    args = ap.parse_args()

    engine = pick_engine(args.engine)
    staging = Path(args.staging) if args.staging else (
        DIST_DIR / "staging" / f"DepthForge-{VERSION}-windows-x86_64-offline"
    )
    out_base = f"DepthForge-{VERSION}-windows-x86_64-setup"

    section(f"Building {out_base}.exe")
    check_payload(staging)

    out = compile_iss(engine, staging, out_base)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (DIST_DIR / f"{out.name}.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    log(f"{out.name}: {human(out.stat().st_size)}")
    log(f"sha256: {digest}")

    section("Done")
    print(f"  {out}  ({human(out.stat().st_size)})")
    print()
    print("  NOTE: cross-built from Linux — nothing here can execute it.")
    print("  Run it on a real Windows machine before announcing the release.")


if __name__ == "__main__":
    main()
