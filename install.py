#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DepthForge – Kompleksowy instalator dla stacji klienta
=======================================================
Uruchomienie (wystarczy systemowy Python 3.8+, bez venv):

    python install.py                  # pełna instalacja
    python install.py --skip-models    # bez pobierania modeli (~685 MB)
    python install.py --skip-gimp      # bez instalacji wtyczki GIMP
    python install.py --gimp-only      # tylko wtyczka GIMP
    python install.py --uninstall      # usuwa wtyczkę i wpisy env

Co robi instalator
------------------
1. Sprawdza wymagania (Python >= 3.8, GIMP 3.x)
2. Tworzy wirtualne środowisko .venv w katalogu projektu
3. Instaluje wszystkie zależności Python (requirements.txt)
4. Pobiera modele OpenVINO (DPT Large + MiDaS v2.1 Small) – łącznie ~685 MB
5. Instaluje wtyczkę GIMP 3.x
6. Ustawia zmienną środowiskową DEPTHFORGE_PYTHON (dla bieżącego użytkownika)
7. Zapisuje plik konfiguracyjny obok wtyczki GIMP
"""

import sys
import os
import subprocess
import shutil
import platform
import argparse
import json
import urllib.request
import urllib.error
import struct
import zipfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Kolory ANSI (Windows 10+)
# ─────────────────────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    os.system("")   # włącz sekwencje ANSI w cmd/PowerShell

_R = "\033[31m";  _G = "\033[32m";  _Y = "\033[33m"
_B = "\033[34m";  _C = "\033[36m";  _W = "\033[0m";  _BOLD = "\033[1m"

def ok(msg):    print(f"  {_G}✓{_W}  {msg}")
def err(msg):   print(f"  {_R}✗{_W}  {msg}")
def warn(msg):  print(f"  {_Y}⚠{_W}  {msg}")
def info(msg):  print(f"  {_C}→{_W}  {msg}")
def step(msg):  print(f"\n{_BOLD}{_B}{'─'*60}{_W}\n{_BOLD}  {msg}{_W}\n{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  Konfiguracja projektu
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
VENV_DIR       = PROJECT_ROOT / ".venv"
REQUIREMENTS   = PROJECT_ROOT / "requirements.txt"
PLUGIN_SRC     = PROJECT_ROOT / "gimp_plugins" / "depthforge"
DOWNLOAD_PY    = PROJECT_ROOT / "download_models.py"

GITHUB_REPO    = "GrzegorzOle/DepthForge"
DEFAULT_RELEASE = "v0.1.0"

MODELS = [
    {
        "name":       "DPT Large – weights (.bin)",
        "filename":   "dpt_large.bin",
        "local_path": PROJECT_ROOT / "models" / "dpt" / "openvino" / "dpt_large.bin",
        "size_mb":    652,
    },
    {
        "name":       "DPT Large – graph (.xml)",
        "filename":   "dpt_large.xml",
        "local_path": PROJECT_ROOT / "models" / "dpt" / "openvino" / "dpt_large.xml",
        "size_mb":    1,
    },
    {
        "name":       "MiDaS v2.1 Small – weights (.bin)",
        "filename":   "midas_v21_small_256.bin",
        "local_path": PROJECT_ROOT / "models" / "midas" / "openvino" / "midas_v21_small_256.bin",
        "size_mb":    32,
    },
    {
        "name":       "MiDaS v2.1 Small – graph (.xml)",
        "filename":   "midas_v21_small_256.xml",
        "local_path": PROJECT_ROOT / "models" / "midas" / "openvino" / "midas_v21_small_256.xml",
        "size_mb":    1,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Lokalizacja GIMP
# ─────────────────────────────────────────────────────────────────────────────
def find_gimp_install() -> Path | None:
    """Znajdź katalog instalacji GIMP 3.x (Windows / Linux / macOS)."""
    system = platform.system()
    if system == "Windows":
        # Rejestr – instalacja dla użytkownika
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for arch in ("SOFTWARE", r"SOFTWARE\WOW6432Node"):
                try:
                    key_path = rf"{arch}\Microsoft\Windows\CurrentVersion\Uninstall"
                    with winreg.OpenKey(hive, key_path) as base:
                        i = 0
                        while True:
                            try:
                                sub = winreg.EnumKey(base, i)
                                with winreg.OpenKey(base, sub) as k:
                                    name = winreg.QueryValueEx(k, "DisplayName")[0]
                                    if "GIMP" in name and "3." in name:
                                        loc = winreg.QueryValueEx(k, "InstallLocation")[0]
                                        if loc:
                                            return Path(loc)
                            except OSError:
                                break
                            i += 1
                except OSError:
                    continue
        # Znany domyślny katalog
        fallback = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GIMP 3"
        if fallback.exists():
            return fallback
    elif system == "Darwin":
        for p in ["/Applications/GIMP-3.app", "/Applications/GIMP.app"]:
            if Path(p).exists():
                return Path(p)
    else:
        for p in ["/usr/bin/gimp", "/usr/local/bin/gimp"]:
            if Path(p).exists():
                return Path(p).parent.parent
    return None


def find_gimp_plugin_dir() -> Path | None:
    """
    Return the plug-ins directory for the current user.
    GIMP 3.2.x uses %APPDATA%\\GIMP\\3.2\\plug-ins\\
    GIMP 3.0.x uses %APPDATA%\\GIMP\\3.0\\plug-ins\\
    We detect which sub-directory actually exists, preferring the highest version.
    """
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", ""))
        gimp_base = base / "GIMP"
    elif system == "Darwin":
        gimp_base = Path.home() / "Library" / "Application Support" / "GIMP"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        gimp_base = Path(xdg) / "GIMP"

    if not gimp_base.exists():
        # Fall back to 3.0 default
        return gimp_base / "3.0" / "plug-ins"

    # Find the highest existing version directory
    candidates = sorted(
        [d for d in gimp_base.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: [int(x) for x in d.name.split(".") if x.isdigit()],
        reverse=True,
    )
    if candidates:
        return candidates[0] / "plug-ins"

    return gimp_base / "3.0" / "plug-ins"


# ─────────────────────────────────────────────────────────────────────────────
#  Krok 1 – Sprawdzenie wymagań
# ─────────────────────────────────────────────────────────────────────────────
def check_requirements() -> bool:
    step("KROK 1 – Sprawdzenie wymagań systemowych")
    all_ok = True

    # Python
    v = sys.version_info
    if v >= (3, 8):
        ok(f"Python {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    else:
        err(f"Python {v.major}.{v.minor} – wymagany >= 3.8")
        all_ok = False

    # GIMP
    gimp_dir = find_gimp_install()
    if gimp_dir:
        ok(f"GIMP znaleziony: {gimp_dir}")
    else:
        warn("GIMP 3.x nie znaleziony – wtyczka nie zostanie zainstalowana")
        warn("Pobierz GIMP 3.x z: https://www.gimp.org/downloads/")

    # pip (powinien być dostępny w standardowym Python)
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       check=True, capture_output=True)
        ok("pip dostępny")
    except subprocess.CalledProcessError:
        err("pip niedostępny – uruchom:  python -m ensurepip --upgrade")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  Wykrywanie najlepszego Pythona do stworzenia venv
# ─────────────────────────────────────────────────────────────────────────────
# numpy/scipy/opencv mają gotowe binary wheels dla Python 3.10-3.12.
# Python 3.13+ wymaga kompilacji ze źródeł (brak wheels → błąd instalacji).
_PREFERRED_VERSIONS = [(3, 12), (3, 11), (3, 10), (3, 13), (3, 9)]


def _get_python_version(exe: str) -> tuple | None:
    """Zwróć (major, minor) dla podanego interpretera lub None."""
    try:
        r = subprocess.run(
            [exe, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split()
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


def find_best_python() -> str:
    """
    Znajdź najlepszy interpreter Python spośród dostępnych w systemie.
    Preferuje wersje 3.12 → 3.11 → 3.10, bo mają gotowe binary wheels
    dla numpy, scipy, opencv i openvino.
    """
    candidates = []

    def _add(exe: str):
        if exe and os.path.isfile(exe):
            ver = _get_python_version(exe)
            if ver and ver >= (3, 8):
                candidates.append((ver, exe))

    # --- Windows-specific locations ------------------------------------------
    if platform.system() == "Windows":
        home = Path.home()

        # uv / pyenv / mise – ~/.local/bin/python3.XX.exe
        local_bin = home / ".local" / "bin"
        if local_bin.exists():
            for exe in sorted(local_bin.glob("python3.*.exe"), reverse=True):
                _add(str(exe))

        # Standard Python.org installer – AppData\Local\Programs\Python\Python3XX
        local_prog = home / "AppData" / "Local" / "Programs" / "Python"
        if local_prog.exists():
            for d in sorted(local_prog.iterdir(), reverse=True):
                _add(str(d / "python.exe"))

        # chocolatey – C:\ProgramData\chocolatey\bin\python3.XX.exe
        choco = Path(r"C:\ProgramData\chocolatey\bin")
        if choco.exists():
            for exe in sorted(choco.glob("python3.*.exe"), reverse=True):
                _add(str(exe))

        # System-wide C:\Program Files\Python3XX
        for d in sorted(Path(r"C:\Program Files").glob("Python3*"), reverse=True):
            _add(str(d / "python.exe"))

    # --- PATH search (all platforms) -----------------------------------------
    for name in ("python3.12", "python3.11", "python3.10", "python3.13",
                 "python3.9", "python3", "python"):
        found = shutil.which(name)
        if found:
            _add(found)

    # Current interpreter base (in case we're already in the right venv)
    if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        base_py = Path(sys.base_prefix) / (
            "python.exe" if platform.system() == "Windows" else "bin/python3"
        )
        _add(str(base_py))
    else:
        _add(sys.executable)

    if not candidates:
        return sys.executable

    def preference(item):
        ver, _ = item
        try:
            return _PREFERRED_VERSIONS.index(ver)
        except ValueError:
            return 99

    candidates.sort(key=preference)
    # Deduplicate by real path
    seen, unique = set(), []
    for ver, exe in candidates:
        real = os.path.realpath(exe)
        if real not in seen:
            seen.add(real)
            unique.append((ver, exe))

    best_ver, best_exe = unique[0]
    info(f"Dostępne Pythony: {', '.join(f'{v[0]}.{v[1]}' for v,_ in unique[:5])}")
    return best_exe


# ─────────────────────────────────────────────────────────────────────────────
#  Krok 2 – Wirtualne środowisko Python
# ─────────────────────────────────────────────────────────────────────────────
def setup_venv(yes: bool = False) -> Path | None:
    step("KROK 2 – Wirtualne środowisko Python (.venv)")

    # Wybierz najlepszy Python bazowy
    base_python = find_best_python()
    base_ver    = _get_python_version(base_python)
    info(f"Python bazowy: {base_python}  (wersja {base_ver[0]}.{base_ver[1]})")

    if base_ver and base_ver >= (3, 13):
        warn(f"Python {base_ver[0]}.{base_ver[1]} – numpy może nie mieć gotowych wheels.")
        warn("Instalacja może potrwać dłużej lub wymagać narzędzi kompilacji.")
        warn("Zalecana wersja: Python 3.12 lub 3.11")

    if VENV_DIR.exists():
        # Sprawdź czy venv ma właściwą wersję
        if platform.system() == "Windows":
            existing_py = VENV_DIR / "Scripts" / "python.exe"
        else:
            existing_py = VENV_DIR / "bin" / "python3"
        existing_ver = _get_python_version(str(existing_py)) if existing_py.exists() else None

        if existing_ver and base_ver and existing_ver != base_ver:
            warn(f"Venv ma Python {existing_ver[0]}.{existing_ver[1]}, "
                 f"ale preferowany jest {base_ver[0]}.{base_ver[1]}.")
            if yes:
                ans = "t"
            else:
                ans = input("  Usunąć stary venv i utworzyć nowy? [t/N] ").strip().lower()
            if ans in ("t", "y", "tak", "yes"):
                shutil.rmtree(VENV_DIR)
                info("Stary venv usunięty.")
            else:
                ok(f"Używam istniejącego venv (Python {existing_ver[0]}.{existing_ver[1]})")
        else:
            ok(f"Venv już istnieje: {VENV_DIR}")

    if not VENV_DIR.exists():
        info(f"Tworzę venv w {VENV_DIR} …")
        r = subprocess.run(
            [base_python, "-m", "venv", str(VENV_DIR)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            err(f"Nie udało się utworzyć venv:\n{r.stderr}")
            return None
        ok("Venv utworzony")

    # Ścieżka do Pythona w venv
    if platform.system() == "Windows":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python3"
        if not venv_python.exists():
            venv_python = VENV_DIR / "bin" / "python"

    if not venv_python.exists():
        err(f"Nie znaleziono Pythona w venv: {venv_python}")
        return None

    ver = _get_python_version(str(venv_python))
    ok(f"Python venv: {venv_python}  ({ver[0]}.{ver[1] if ver else '?'})")
    return venv_python


# ─────────────────────────────────────────────────────────────────────────────
#  Krok 3 – Instalacja zależności
# ─────────────────────────────────────────────────────────────────────────────
def install_dependencies(venv_python: Path) -> bool:
    step("KROK 3 – Instalacja zależności Python")

    if not REQUIREMENTS.exists():
        err(f"Brak pliku {REQUIREMENTS}")
        return False

    info("Aktualizacja pip …")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=False,
    )

    info(f"Instalacja pakietów z {REQUIREMENTS.name} …")
    info("(Może to potrwać kilka minut przy pierwszej instalacji)")
    r = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        err(f"Błąd instalacji zależności:")
        print(r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr)
        return False

    ok("Wszystkie zależności zainstalowane")

    # Weryfikacja kluczowych pakietów
    for pkg in ("numpy", "cv2", "openvino"):
        test = subprocess.run(
            [str(venv_python), "-c", f"import {pkg}; print({pkg}.__version__ if hasattr({pkg}, '__version__') else 'ok')"],
            capture_output=True, text=True,
        )
        if test.returncode == 0:
            ok(f"  {pkg}: {test.stdout.strip()}")
        else:
            warn(f"  {pkg}: nie załadowany ({test.stderr.strip()[:80]})")

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Krok 4 – Pobieranie modeli OpenVINO
# ─────────────────────────────────────────────────────────────────────────────
def _progress_bar(block_num: int, block_size: int, total_size: int):
    if total_size <= 0:
        return
    dl   = min(block_num * block_size, total_size)
    pct  = dl / total_size * 100
    bar  = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
    mb   = dl / 1_048_576
    tot  = total_size / 1_048_576
    print(f"\r    [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)


def download_models(release: str = DEFAULT_RELEASE) -> bool:
    step("KROK 4 – Pobieranie modeli OpenVINO")
    info(f"Release: {release}  (GitHub: {GITHUB_REPO})")

    all_ok = True
    for model in MODELS:
        dest: Path = model["local_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            if size_mb >= model["size_mb"] * 0.9:
                ok(f"Już pobrano: {dest.name}  ({size_mb:.0f} MB)")
                continue
            warn(f"Niekompletny: {dest.name} – pobieranie ponownie …")

        url = (f"https://github.com/{GITHUB_REPO}/releases/download"
               f"/{release}/{model['filename']}")
        info(f"Pobieranie: {model['name']}")
        print(f"    URL: {url}")

        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_bar)
            print()
            ok(f"{dest.name}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
        except urllib.error.HTTPError as e:
            print()
            err(f"HTTP {e.code}: {e.reason}")
            if e.code == 404:
                warn(f"Release '{release}' nie istnieje lub brak pliku.")
                warn(f"Sprawdź: https://github.com/{GITHUB_REPO}/releases")
            all_ok = False
        except urllib.error.URLError as e:
            print()
            err(f"Błąd sieci: {e.reason}")
            all_ok = False
        except KeyboardInterrupt:
            if dest.exists():
                dest.unlink()
            print("\nPrzerwano przez użytkownika.")
            sys.exit(1)

    if not all_ok:
        warn("Niektóre modele nie zostały pobrane.")
        warn("Wtyczka nadal będzie działać używając trybu syntetycznego (fallback).")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  Krok 5 – Instalacja wtyczki GIMP
# ─────────────────────────────────────────────────────────────────────────────
def install_gimp_plugin(venv_python: Path) -> bool:
    step("KROK 5 – Instalacja wtyczki GIMP 3.x")

    plugin_dir = find_gimp_plugin_dir()
    if not plugin_dir:
        err("Nie można ustalić katalogu plug-ins GIMP")
        return False

    gimp_install = find_gimp_install()
    if not gimp_install:
        warn("GIMP 3.x nie znaleziony – pomiń instalację wtyczki.")
        warn("Aby zainstalować później: python gimp_plugins\\install_plugin.py")
        return False

    info(f"GIMP:       {gimp_install}")
    info(f"plug-ins:   {plugin_dir}")
    info(f"Źródło:     {PLUGIN_SRC}")

    if not PLUGIN_SRC.exists():
        err(f"Brak folderu wtyczki: {PLUGIN_SRC}")
        return False

    # ── Usuń starą wersję z katalogu SYSTEMOWEGO GIMP (jeśli istnieje) ───────
    system_plugin = gimp_install / "lib" / "gimp" / "3.0" / "plug-ins" / "depthforge"
    if system_plugin.exists():
        info(f"Usuwam starą wersję z katalogu systemowego: {system_plugin}")
        try:
            shutil.rmtree(system_plugin)
            ok("Stara wersja systemowa usunięta")
        except Exception as e:
            warn(f"Nie można usunąć systemowej wersji: {e}")

    dest = plugin_dir / "depthforge"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        info("Usuwam starą wersję wtyczki …")
        shutil.rmtree(dest)

    shutil.copytree(PLUGIN_SRC, dest)
    ok(f"Wtyczka skopiowana → {dest}")

    # Na Unix nadaj prawa wykonania
    if platform.system() != "Windows":
        main_script = dest / "depthforge.py"
        if main_script.exists():
            os.chmod(main_script, 0o755)
            ok("chmod +x depthforge.py")

    # Zapisz plik konfiguracyjny obok wtyczki
    config_path = dest / "depthforge_install.json"
    config = {
        "project_root":  str(PROJECT_ROOT),
        "venv_python":   str(venv_python),
        "installed_by":  "install.py",
        "version":       "0.1.0",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    ok(f"Konfiguracja zapisana: {config_path.name}")

    # Ustaw zmienną środowiskową DEPTHFORGE_PYTHON dla użytkownika
    _set_user_env("DEPTHFORGE_PYTHON", str(venv_python))

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Ustawianie zmiennej środowiskowej użytkownika
# ─────────────────────────────────────────────────────────────────────────────
def _set_user_env(name: str, value: str):
    """Ustaw zmienną środowiskową na poziomie bieżącego użytkownika (trwale)."""
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, "Environment",
                0, winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
            winreg.CloseKey(key)
            # Powiadom system o zmianie środowiska
            import ctypes
            ctypes.windll.user32.SendMessageTimeoutW(
                0xFFFF, 0x001A, 0, "Environment", 2, 5000, None
            )
            ok(f"Zmienna env ustawiona: {name} = {value}")
        except Exception as e:
            warn(f"Nie udało się ustawić {name} w rejestrze: {e}")
            warn(f"Ustaw ręcznie: setx {name} \"{value}\"")
    else:
        # Linux/macOS – dописz do ~/.profile i ~/.bashrc
        line = f'\nexport {name}="{value}"\n'
        for rc in (Path.home() / ".profile", Path.home() / ".bashrc"):
            try:
                content = rc.read_text(encoding="utf-8") if rc.exists() else ""
                if name not in content:
                    with open(rc, "a", encoding="utf-8") as f:
                        f.write(line)
                    ok(f"Dodano do {rc.name}: export {name}")
            except Exception:
                pass


def _remove_user_env(name: str):
    """Usuń zmienną środowiskową użytkownika."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, "Environment",
                0, winreg.KEY_SET_VALUE
            )
            try:
                winreg.DeleteValue(key, name)
                ok(f"Usunięto zmienną: {name}")
            except FileNotFoundError:
                pass
            winreg.CloseKey(key)
        except Exception as e:
            warn(f"Błąd usuwania {name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Odinstalowanie
# ─────────────────────────────────────────────────────────────────────────────
def uninstall():
    step("ODINSTALOWANIE DepthForge GIMP plugin")

    plugin_dir = find_gimp_plugin_dir()
    if plugin_dir:
        dest = plugin_dir / "depthforge"
        if dest.exists():
            shutil.rmtree(dest)
            ok(f"Usunięto: {dest}")
        else:
            info("Wtyczka nie była zainstalowana")

    _remove_user_env("DEPTHFORGE_PYTHON")
    ok("Gotowe")


# ─────────────────────────────────────────────────────────────────────────────
#  Podsumowanie
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(venv_python: Path | None):
    step("PODSUMOWANIE INSTALACJI")

    gimp_plugin = find_gimp_plugin_dir()
    if gimp_plugin:
        plugin_dest = gimp_plugin / "depthforge"
        if plugin_dest.exists():
            ok(f"Wtyczka GIMP: {plugin_dest}")

    if venv_python and venv_python.exists():
        ok(f"Python venv:  {venv_python}")

    env_val = os.environ.get("DEPTHFORGE_PYTHON", "")
    if env_val:
        ok(f"DEPTHFORGE_PYTHON = {env_val}")

    models_ok = all(m["local_path"].exists() for m in MODELS)
    if models_ok:
        ok("Modele OpenVINO: pobrane")
    else:
        missing = [m["name"] for m in MODELS if not m["local_path"].exists()]
        warn(f"Brakujące modele ({len(missing)}): wtyczka użyje trybu syntetycznego")
        for m in missing:
            warn(f"  – {m}")

    print(f"""
{_BOLD}{'='*60}{_W}
{_BOLD}{_G}  INSTALACJA ZAKOŃCZONA!{_W}
{'='*60}

  Jak używać wtyczki:
  1. {_BOLD}Zamknij i uruchom ponownie GIMP{_W}
  2. Otwórz dowolny obraz
  3. Kliknij: {_BOLD}Filtry → DepthForge → Generate Depth Map…{_W}

  Szybki test z linii poleceń:
  {_C}{venv_python or 'python'} src/depth_pipeline.py --input data/sample_input.jpg{_W}

  Jeśli wtyczka nie pojawia się w GIMP:
    • Sprawdź czy GIMP jest w wersji 3.x
    • Zrestartuj GIMP
    • Otwórz: Filtry → Script-Fu → Konsola i wpisz:
      (cadr (gimp-version))
{_BOLD}{'='*60}{_W}
""")


# ─────────────────────────────────────────────────────────────────────────────
#  Główna funkcja
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DepthForge – kompleksowy instalator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python install.py                  # pełna instalacja
  python install.py --skip-models    # bez pobierania modeli (~685 MB)
  python install.py --skip-gimp      # bez instalacji wtyczki GIMP
  python install.py --gimp-only      # tylko wtyczka (venv musi istnieć)
  python install.py --uninstall      # usuwa wtyczkę i zmienne env
  python install.py --release v0.2.0 # instalacja z innego release
        """,
    )
    parser.add_argument("--skip-models",  action="store_true",
                        help="Pomiń pobieranie modeli OpenVINO")
    parser.add_argument("--skip-gimp",    action="store_true",
                        help="Pomiń instalację wtyczki GIMP")
    parser.add_argument("--gimp-only",    action="store_true",
                        help="Zainstaluj tylko wtyczkę GIMP (venv musi istnieć)")
    parser.add_argument("--uninstall",    action="store_true",
                        help="Odinstaluj wtyczkę GIMP")
    parser.add_argument("--yes", "-y",         action="store_true",
                        help="Automatycznie odpowiedz Tak na wszystkie pytania")
    parser.add_argument("--release", default=DEFAULT_RELEASE,
                        help=f"Tag release GitHub (domyslnie: {DEFAULT_RELEASE})")
    args = parser.parse_args()

    print(f"""
{_BOLD}{_C}
  ╔══════════════════════════════════════╗
  ║   DepthForge  –  Instalator          ║
  ║   Projekt: {PROJECT_ROOT!s:<29}║
  ╚══════════════════════════════════════╝
{_W}""")

    # Odinstalowanie
    if args.uninstall:
        uninstall()
        return

    # Tylko wtyczka GIMP
    if args.gimp_only:
        if platform.system() == "Windows":
            venv_python = VENV_DIR / "Scripts" / "python.exe"
        else:
            venv_python = VENV_DIR / "bin" / "python3"
        if not venv_python.exists():
            err("Venv nie istnieje. Uruchom najpierw pełną instalację.")
            sys.exit(1)
        if install_gimp_plugin(venv_python):
            print_summary(venv_python)
        return

    # ── Pełna instalacja ──────────────────────────────────────────────────────

    # 1. Sprawdzenie wymagań
    if not check_requirements():
        err("Wymagania nie są spełnione. Przerywam instalację.")
        sys.exit(1)

    # 2. Venv
    venv_python = setup_venv(yes=args.yes)
    if not venv_python:
        err("Nie udało się skonfigurować środowiska Python.")
        sys.exit(1)

    # 3. Zależności
    if not install_dependencies(venv_python):
        err("Błąd instalacji zależności.")
        sys.exit(1)

    # 4. Modele
    if not args.skip_models:
        download_models(args.release)
    else:
        warn("Pominięto pobieranie modeli (--skip-models).")
        warn("Wtyczka użyje syntetycznego trybu fallback.")

    # 5. Wtyczka GIMP
    if not args.skip_gimp:
        install_gimp_plugin(venv_python)
    else:
        warn("Pominięto instalację wtyczki GIMP (--skip-gimp).")

    # Podsumowanie
    print_summary(venv_python)


if __name__ == "__main__":
    main()

