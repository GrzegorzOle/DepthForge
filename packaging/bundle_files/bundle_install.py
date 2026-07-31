#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – bundle installer
=============================
Runs under the CPython interpreter shipped inside this bundle, never under a
system Python and never under GIMP's. It:

  1. detects the GIMP that is installed (uninstall registry on Windows,
     `gimp --version` elsewhere) and derives its user plug-ins directory,
  2. copies the depthforge plugin there,
  3. writes depthforge_install.json pointing the plugin at this bundle's
     interpreter and app directory,
  4. clears a stale DEPTHFORGE_PYTHON left over by an older source install,
     which would otherwise keep pointing GIMP somewhere else,
  5. downloads the OpenVINO models (unless already bundled/present),
  6. verifies that the whole chain actually imports and runs.

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

# The plugin targets the GIMP 3.x API and is tested against 3.2.x. Older 3.0 is
# untested, 2.x cannot load it at all.
MIN_GIMP = (3, 2)


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


def child_env() -> dict:
    """Environment for the interpreters we spawn.

    A globally set PYTHONHOME/PYTHONPATH - left by an old Anaconda, a py
    launcher tweak, or an embedded runtime - makes our self-contained
    interpreter load a foreign stdlib and fail while importing numpy's C
    extensions. Strip them so the bundle is unaffected by whatever Python
    history the machine has.
    """
    env = dict(os.environ)
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(var, None)
    env.setdefault("PYTHONIOENCODING", ":replace")
    return env


# ── GIMP discovery ───────────────────────────────────────────────────────────

def gimp_config_base() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "GIMP"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "GIMP"
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "GIMP"


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


def _detect_gimp_windows() -> tuple[tuple[int, ...] | None, str | None]:
    try:
        import winreg
    except ImportError:
        return None, None

    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    best_ver: tuple[int, ...] | None = None
    best_loc: str | None = None

    for hive, path in roots:
        try:
            root = winreg.OpenKey(hive, path)
        except OSError:
            continue
        with root:
            count = winreg.QueryInfoKey(root)[0]
            for i in range(count):
                try:
                    with winreg.OpenKey(root, winreg.EnumKey(root, i)) as k:
                        name = str(winreg.QueryValueEx(k, "DisplayName")[0])
                        if not name.upper().startswith("GIMP"):
                            continue
                        try:
                            ver = parse_version(winreg.QueryValueEx(k, "DisplayVersion")[0])
                        except OSError:
                            ver = None
                        try:
                            loc = str(winreg.QueryValueEx(k, "InstallLocation")[0]) or None
                        except OSError:
                            loc = None
                except OSError:
                    continue
                if ver and (best_ver is None or ver > best_ver):
                    best_ver, best_loc = ver, loc

    if best_ver:
        return best_ver, best_loc

    # Portable copies and installs whose uninstall entry we cannot read still
    # live in one of the two usual places. The folder name carries the major
    # version only, so the version stays unknown - the location alone is already
    # enough to say "GIMP is here".
    for base in (os.environ.get("LOCALAPPDATA", "") and
                 os.path.join(os.environ["LOCALAPPDATA"], "Programs"),
                 os.environ.get("ProgramFiles", ""),
                 os.environ.get("ProgramFiles(x86)", "")):
        if not base or not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), reverse=True):
            if entry.upper().startswith("GIMP") and \
                    os.path.isdir(os.path.join(base, entry, "bin")):
                return None, os.path.join(base, entry)
    return None, None


def _detect_gimp_posix() -> tuple[tuple[int, ...] | None, str | None]:
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


def detect_installed_gimp() -> tuple[tuple[int, ...] | None, str | None]:
    """Find the GIMP that is *installed*, not the config dir it leaves behind.

    The config directory only appears once GIMP has been launched, so on a fresh
    machine - exactly the machine this installer targets - it says nothing. Ask
    the system what is installed instead. Getting this wrong is silent and
    expensive: dropping the plugin into 3.2/ when the user runs GIMP 3.0 leaves
    a complete, working install that simply never shows up under Filters.

    Returns (version, install location); either part may be None.
    """
    if platform.system() == "Windows":
        return _detect_gimp_windows()
    return _detect_gimp_posix()


def find_gimp_plugin_dir(explicit: str | None,
                         installed: tuple[int, ...] | None
                         ) -> tuple[Path, bool, tuple[int, ...] | None]:
    """Return (plug-ins dir, existed_already, config series we targeted).

    GIMP keeps its user config in <base>/<major>.<minor>/. We aim at the series
    of the GIMP that is actually installed, creating that directory when GIMP
    has never been launched. Only when nothing is installed (or we could not
    tell) do we fall back to the newest config directory present, then to 3.2.
    """
    if explicit:
        return Path(explicit).expanduser() / "plug-ins", True, None

    base = gimp_config_base()
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
            return existing[want] / "plug-ins", True, want
        name = ".".join(str(x) for x in want)
        return base / name / "plug-ins", False, want

    if existing:
        top = max(existing)
        return existing[top] / "plug-ins", True, top

    return base / "3.2" / "plug-ins", False, None


def report_gimp(installed: tuple[int, ...] | None, location: str | None,
                existed: bool, series: tuple[int, ...] | None) -> bool:
    """Say what we found and whether it is good enough. Returns "GIMP is here".

    Warns rather than aborts: installing DepthForge *before* GIMP is a supported
    flow, which the .iss covers with its "Install GIMP plugin" Start menu
    shortcut. The return value only decides whether the caller tells the user to
    use it.
    """
    want = ".".join(str(x) for x in MIN_GIMP)
    found_it = bool(installed or location or existed)

    if installed:
        ok(f"GIMP {'.'.join(str(x) for x in installed)} installed"
           + (f": {location}" if location else ""))
        if installed[:2] < MIN_GIMP:
            warn(f"this plugin needs GIMP {want} or newer - installing anyway,")
            warn("but GIMP will most likely not show it under Filters.")
    elif location:
        ok(f"GIMP found: {location} (version unreadable)")
    elif existed:
        shown = ".".join(str(x) for x in series) if series else "?"
        ok(f"no GIMP installation detected, but a GIMP {shown} config exists")
    else:
        warn("no GIMP found on this computer.")
        warn("Installing the files anyway and assuming GIMP 3.2 - that is the")
        warn("right thing to do if you are installing DepthForge first.")

    if not existed and found_it:
        info("GIMP has never been started, so its config directory is being")
        info("created now. That is normal on a fresh installation.")
    return found_it


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


def clear_stale_env_override(expected: Path) -> None:
    """Drop a DEPTHFORGE_PYTHON left behind by an older source install.

    That variable is machine-wide, so it outlives the install that wrote it: a
    developer install from months ago kept pointing GIMP at a venv that no
    longer worked, and this bundle - correct in every other respect - was
    silently outranked. The plugin now prefers depthforge_install.json, but the
    variable still misleads the CLI, the diagnostics and the next reader of the
    log, so clear it whenever it disagrees with what we just installed.

    Never touched when it already matches us: that is somebody's deliberate
    setup, not a leftover.
    """
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                                winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
                try:
                    current, _ = winreg.QueryValueEx(key, "DEPTHFORGE_PYTHON")
                except FileNotFoundError:
                    return
                if os.path.normcase(os.path.expandvars(str(current))) == \
                        os.path.normcase(str(expected)):
                    ok("DEPTHFORGE_PYTHON already points here - left alone")
                    return
                winreg.DeleteValue(key, "DEPTHFORGE_PYTHON")
            info(f"removed stale DEPTHFORGE_PYTHON = {current}")
            info("(it pointed at a different interpreter, from an older install)")
            # Without the broadcast, Explorer - and therefore GIMP - keeps the
            # old environment block until the user logs out.
            import ctypes
            ctypes.windll.user32.SendMessageTimeoutW(
                0xFFFF, 0x001A, 0, "Environment", 2, 5000, None
            )
        except Exception as e:
            warn(f"could not check DEPTHFORGE_PYTHON in the registry: {e}")
        return

    # POSIX: the old install.py appended `export DEPTHFORGE_PYTHON=...` to the
    # shell rc files. Those lines are ours and nothing else writes them.
    for rc in (Path.home() / ".profile", Path.home() / ".bashrc"):
        try:
            if not rc.is_file():
                continue
            lines = rc.read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [ln for ln in lines
                    if not ln.lstrip().startswith("export DEPTHFORGE_PYTHON=")]
            if len(kept) != len(lines):
                rc.write_text("".join(kept), encoding="utf-8")
                info(f"removed stale DEPTHFORGE_PYTHON export from {rc}")
        except Exception as e:
            warn(f"could not clean {rc}: {e}")

    current = os.environ.get("DEPTHFORGE_PYTHON", "")
    if current and os.path.realpath(current) != os.path.realpath(str(expected)):
        warn(f"DEPTHFORGE_PYTHON={current} is still set in this shell.")
        warn("Start GIMP from a new shell, or log out, so it does not inherit it.")


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
        cwd=str(app_root), env=child_env(),
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
        env=child_env(),
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

    installed_ver, installed_loc = detect_installed_gimp()
    plugin_dir, existed, series = find_gimp_plugin_dir(args.gimp_dir, installed_ver)

    if args.uninstall:
        section("Uninstalling")
        uninstall(plugin_dir)
        return

    section("GIMP")
    gimp_found = report_gimp(installed_ver, installed_loc, existed, series)
    info(f"plug-ins directory: {plugin_dir}")

    section("Plugin")
    dest = install_plugin(plugin_dir)
    write_install_json(dest, app_root)
    clear_stale_env_override(bundled_python())

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

    # Everything shipped and verified, but there is no GIMP to receive it. Exit
    # non-zero so the .exe surfaces its "install the plugin later" dialog instead
    # of reporting a clean success the user cannot act on.
    if not gimp_found and not args.gimp_dir:
        section("GIMP not found")
        print("  DepthForge itself is installed and working, but no GIMP was")
        print("  found on this computer, so the plugin has nowhere to appear.")
        print()
        print("  Install GIMP 3.2 or newer, then run this installer again -")
        print('  on Windows via the Start menu shortcut "Install GIMP plugin".')
        sys.exit(2)


if __name__ == "__main__":
    main()
