#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – GIMP 3.x Plugin Installer
=======================================
Copies the 'depthforge' plugin folder to GIMP's plug-ins directory and
optionally installs the required Python dependencies (numpy, opencv-python).

Usage
-----
    python gimp_plugins/install_plugin.py            # auto-detect GIMP path
    python gimp_plugins/install_plugin.py --gimp-dir "C:\\path\\to\\GIMP\\3.0"
    python gimp_plugins/install_plugin.py --uninstall

Compatible with: GIMP 3.2.x on Windows / Linux / macOS
"""

import sys
import os
import shutil
import argparse
import platform
import subprocess


# ──────────────────────────────────────────────────────────────────────────────
#  Platform-specific default plug-ins directory
# ──────────────────────────────────────────────────────────────────────────────
def default_gimp_plugin_dir() -> str | None:
    """
    Return the default GIMP 3.x user plug-ins directory.
    GIMP 3.2.x uses %APPDATA%\\GIMP\\3.2\\  (not 3.0\\)
    We detect which version dir exists, preferring the highest.
    """
    import os, platform
    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        gimp_base = os.path.join(appdata, "GIMP")
    elif system == "Darwin":
        home = os.path.expanduser("~")
        gimp_base = os.path.join(home, "Library", "Application Support", "GIMP")
    else:
        home = os.path.expanduser("~")
        xdg = os.environ.get("XDG_CONFIG_HOME", os.path.join(home, ".config"))
        gimp_base = os.path.join(xdg, "GIMP")

    if not os.path.isdir(gimp_base):
        return os.path.join(gimp_base, "3.0", "plug-ins")

    # Find highest version directory
    try:
        dirs = [
            d for d in os.listdir(gimp_base)
            if os.path.isdir(os.path.join(gimp_base, d)) and d[0].isdigit()
        ]
        dirs.sort(key=lambda d: [int(x) for x in d.split(".") if x.isdigit()], reverse=True)
        if dirs:
            return os.path.join(gimp_base, dirs[0], "plug-ins")
    except Exception:
        pass

    return os.path.join(gimp_base, "3.0", "plug-ins")


# ──────────────────────────────────────────────────────────────────────────────
#  Install / uninstall helpers
# ──────────────────────────────────────────────────────────────────────────────
PLUGIN_FOLDER_NAME = "depthforge"


def _find_venv_python(project_root: str) -> str:
    """Return path to venv Python inside project_root, or empty string."""
    candidates = [
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, ".venv", "bin",     "python3"),
        os.path.join(project_root, ".venv", "bin",     "python"),
        os.path.join(project_root, "venv",  "Scripts", "python.exe"),
        os.path.join(project_root, "venv",  "bin",     "python3"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def _write_install_json(dest_plugin_dir: str, project_root: str) -> None:
    """Write depthforge_install.json so the plugin knows where the project lives."""
    import json
    venv_py = _find_venv_python(project_root)
    data = {
        "project_root": project_root,
        "venv_python":  venv_py,
    }
    json_path = os.path.join(dest_plugin_dir, "depthforge_install.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"depthforge_install.json written:")
    print(f"  project_root = {project_root}")
    print(f"  venv_python  = {venv_py or '(not found)'}")


def install(plugin_src: str, gimp_plugin_dir: str) -> bool:
    dest = os.path.join(gimp_plugin_dir, PLUGIN_FOLDER_NAME)

    print(f"Source  : {plugin_src}")
    print(f"Dest    : {dest}")

    # Create destination parent if needed
    os.makedirs(gimp_plugin_dir, exist_ok=True)

    # Remove old version
    if os.path.exists(dest):
        print("Removing old installation…")
        shutil.rmtree(dest)

    # Copy plugin folder
    shutil.copytree(plugin_src, dest)
    print(f"Plugin copied → {dest}")

    # On Unix: make the script executable
    if platform.system() != "Windows":
        main_script = os.path.join(dest, "depthforge.py")
        if os.path.isfile(main_script):
            os.chmod(main_script, 0o755)
            print(f"chmod +x  {main_script}")

    # Write install JSON so the plugin can find the project root and venv Python
    # project_root = two levels up from this script (gimp_plugins/../ = project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _write_install_json(dest, project_root)

    return True


def uninstall(gimp_plugin_dir: str) -> bool:
    dest = os.path.join(gimp_plugin_dir, PLUGIN_FOLDER_NAME)
    if os.path.exists(dest):
        shutil.rmtree(dest)
        print(f"Uninstalled: {dest}")
    else:
        print(f"Nothing to remove at {dest}")
    return True


def install_python_deps():
    """Install numpy and opencv-python into the current Python environment."""
    deps = ["numpy", "opencv-python"]
    print(f"\nInstalling Python dependencies: {', '.join(deps)}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + deps
    result = subprocess.run(cmd)
    return result.returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DepthForge GIMP 3.x Plugin Installer"
    )
    parser.add_argument(
        "--gimp-dir",
        default=None,
        help="Path to GIMP 3.0 user directory (e.g. %%APPDATA%%\\GIMP\\3.0). "
             "Defaults to the OS-specific location.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the plugin from GIMP's plug-ins directory.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Also install numpy and opencv-python via pip.",
    )
    args = parser.parse_args()

    # Determine the source plugin folder (sibling of this script)
    here       = os.path.dirname(os.path.abspath(__file__))
    plugin_src = os.path.join(here, PLUGIN_FOLDER_NAME)

    if not os.path.isdir(plugin_src):
        print(f"ERROR: Plugin source folder not found: {plugin_src}")
        sys.exit(1)

    # Determine the GIMP plug-ins directory
    if args.gimp_dir:
        gimp_plugin_dir = os.path.join(args.gimp_dir, "plug-ins")
    else:
        gimp_plugin_dir = default_gimp_plugin_dir()
        if gimp_plugin_dir is None:
            print("ERROR: Could not determine GIMP plug-ins directory. "
                  "Use --gimp-dir to specify it manually.")
            sys.exit(1)

    print(f"GIMP plug-ins dir: {gimp_plugin_dir}\n")

    if args.uninstall:
        ok = uninstall(gimp_plugin_dir)
    else:
        ok = install(plugin_src, gimp_plugin_dir)
        if ok and args.install_deps:
            install_python_deps()

        if ok:
            print("\n✓  Installation complete.")
            print("   Restart GIMP and find the plugin under:")
            print("   Filters  →  DepthForge  →  Generate Depth Map…")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

