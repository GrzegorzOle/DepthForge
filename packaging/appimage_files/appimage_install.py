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

def parse_version(text: str) -> tuple[int, ...] | None:
    """Leading dotted integers of a version string: '3.2.4 (git)' -> (3, 2, 4)."""
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    return tuple(parts) if parts else None


def detect_installed_gimp() -> tuple[tuple[int, ...] | None, str | None]:
    """Ask the installed GIMP its version, rather than guessing from ~/.config.

    The config directory only appears once GIMP has been launched, so it says
    nothing about a freshly installed one - and picking the wrong series drops
    the plugin somewhere GIMP will never read, silently.
    """
    exe = shutil.which("gimp-3.0") or shutil.which("gimp-3") or shutil.which("gimp")
    if not exe:
        return None, None
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True,
                           errors="replace", timeout=30, env=child_env())
        for token in (r.stdout or "").split():
            ver = parse_version(token)
            if ver and len(ver) >= 2:
                return ver, exe
    except Exception:
        pass
    return None, exe


def find_gimp_plugin_dir(explicit: str | None,
                         installed: tuple[int, ...] | None) -> tuple[Path, bool]:
    if explicit:
        return Path(explicit).expanduser() / "plug-ins", True

    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    base = Path(xdg) / "GIMP"
    existing: dict[tuple[int, ...], Path] = {}
    if base.is_dir():
        for d in base.iterdir():
            if not d.is_dir():
                continue
            ver = parse_version(d.name)
            if ver and len(ver) >= 2:
                existing[ver[:2]] = d

    if installed:
        want = installed[:2]
        if want in existing:
            return existing[want] / "plug-ins", True
        return base / ".".join(str(x) for x in want) / "plug-ins", False

    if existing:
        return existing[max(existing)] / "plug-ins", True

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


def clear_stale_env_override(expected: Path) -> None:
    """Drop a DEPTHFORGE_PYTHON left behind by an older source install.

    Mirrors bundle_install.py. Older versions of install.py appended an
    ``export DEPTHFORGE_PYTHON=...`` line to the shell rc files; being
    machine-wide, it outlived the install that wrote it and kept pointing GIMP
    at an interpreter that no longer exists. Those lines are ours - nothing else
    writes them - so removing them is safe. Left alone when it already points at
    this AppImage: that is a deliberate setup, not a leftover.
    """
    for rc in (Path.home() / ".profile", Path.home() / ".bashrc"):
        try:
            if not rc.is_file():
                continue
            lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [ln for ln in lines
                    if not ln.lstrip().startswith("export DEPTHFORGE_PYTHON=")
                    or str(expected) in ln]
            if len(kept) != len(lines):
                rc.write_text("".join(kept), encoding="utf-8")
                info(f"removed stale DEPTHFORGE_PYTHON export from {rc}")
        except Exception as e:
            warn(f"could not clean {rc}: {e}")

    current = os.environ.get("DEPTHFORGE_PYTHON", "").strip()
    if current and os.path.realpath(current) != os.path.realpath(str(expected)):
        warn(f"DEPTHFORGE_PYTHON={current} is still set in this shell.")
        warn("Start GIMP from a new shell so it does not inherit the old value.")


def child_env() -> dict:
    """Environment for the interpreters we spawn.

    APPDIR-scoped PYTHONHOME/PYTHONPATH belong to the AppImage runtime, and a
    user's own global ones belong to some other Python entirely; either way the
    child must resolve its own installation. Strip them.
    """
    env = dict(os.environ)
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)
    env.setdefault("PYTHONIOENCODING", ":replace")
    return env


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
        capture_output=True, text=True, errors="replace", cwd=str(app_root),
        env=child_env(),
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

    installed_ver, installed_exe = detect_installed_gimp()
    plugin_dir, existed = find_gimp_plugin_dir(args.gimp_dir, installed_ver)

    if args.uninstall:
        section("Uninstalling")
        uninstall(plugin_dir, args.keep_state)
        return

    section("GIMP")
    if installed_ver:
        ok(f"GIMP {'.'.join(str(x) for x in installed_ver)} installed: {installed_exe}")
    elif installed_exe:
        ok(f"GIMP found: {installed_exe} (version unreadable)")
    elif existed:
        ok("no gimp binary on PATH, but a GIMP config directory exists")
    else:
        warn("no GIMP found on this computer — assuming GIMP 3.2.")
        warn("Install GIMP 3.2+ and re-run with --df-install, or point us at it:")
        warn("  --df-install --gimp-dir <path to your GIMP config dir>")
    info(f"plug-ins directory: {plugin_dir}")
    if not existed and (installed_ver or installed_exe):
        info("GIMP has never been started, so its config directory is being")
        info("created now. That is normal on a fresh installation.")

    section("State directory")
    app_root = build_state_dir()

    section("Plugin")
    dest = install_plugin(plugin_dir)
    write_install_json(dest, app_root)
    clear_stale_env_override(appimage_path())

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
