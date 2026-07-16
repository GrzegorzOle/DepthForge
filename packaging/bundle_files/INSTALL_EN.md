# DepthForge — GIMP plugin installation guide

*(Wersja polska: `INSTALL_PL.md`)*

DepthForge estimates depth from flat 2D images (mainly museum paintings) and
exports STL files for 3D printing — **tactile reproductions read by fingertip**
by blind and partially sighted people.

This package is **fully self-contained**. It ships its own Python together with
every library it needs. **You do not need Python on your system**, and you do
not need to know which Python version your GIMP uses — this package does not
use GIMP's Python at all.

---

## 0. Which file did you download?

DepthForge comes in several shapes. Find yours — the first two do everything for
you, and you can stop reading after this section.

| File | What to do | Models |
|---|---|---|
| `…-setup.exe` | Run it and follow the wizard. It installs the plugin for you. | inside |
| `…-x86_64.AppImage` | `chmod +x` it, then run it. It installs the plugin for you. | inside |
| `…-windows-x86_64.zip` | Unpack, then follow section 2 below. | downloaded during install |
| `…-linux-x86_64.tar.gz` | Unpack, then follow section 2 below. | downloaded during install |

Whichever you picked: **keep the file or folder where it is** after installing.
GIMP will call the Python inside it every time you use the plugin. Moving it is
fine — just run the installer (or the AppImage) once more from the new location.

The AppImage needs **FUSE**, like every AppImage. If it refuses to start with a
message about `libfuse` or `fusermount`, either install your distribution's
FUSE 2 package or use the `.tar.gz` instead.

---

## 1. What you need

| | |
|---|---|
| **GIMP** | version 3.2 or newer |
| **System** | Windows 10/11 64-bit **or** Linux 64-bit (glibc 2.28+: Ubuntu 20.04+, Debian 10+, Fedora 29+) |
| **Disk space** | about 1.5 GB (package + models) |
| **Internet** | only for the `.zip` / `.tar.gz` packages, which download the models (~686 MB) during installation. The `.exe` and `.AppImage` carry the models inside and install fully offline. |
| **RAM** | 4 GB minimum, 8 GB recommended |

Everything runs on the CPU. No graphics card required.

---

## 2. Step-by-step installation

### Step 1 — unpack the package

Extract the archive somewhere it can **stay permanently**:

* **Windows** — right-click the `.zip` → *Extract All…*
  A good location: `C:\DepthForge`
* **Linux** — `tar -xzf DepthForge-0.1.0-linux-x86_64.tar.gz`
  A good location: `~/DepthForge` or `/opt/DepthForge`

> ⚠️ **This matters:** GIMP will reach into this folder for its Python.
> **Do not move or delete the folder after installing.** Don't unpack it into
> *Downloads* if you tend to empty that. If you do need to move it, move the
> folder and then run the installer again from the new location.

### Step 2 — close GIMP

If GIMP is running, close it now. Plugins are only picked up at startup.

### Step 3 — run the installer

* **Windows** — double-click **`install.bat`**

  If you see *"Windows protected your PC"*, click **More info → Run anyway**.
  That is the standard warning for downloaded files that are not code-signed.

* **Linux** — open a terminal in the package folder and run:

  ```bash
  ./install.sh
  ```

The installer will:

1. locate your GIMP plug-ins directory,
2. copy the plugin there,
3. record the path to the Python inside this package,
4. download the models (~686 MB — this can take a few minutes),
5. verify that the whole chain actually works.

It ends with `=== Done ===`. If you see `[FAIL]` instead, see
**Troubleshooting** below.

### Step 4 — start GIMP

You'll find the plugin under:

```
Filters  →  DepthForge  →  Generate Depth Map…
```

---

## 3. First run

1. Open an image in GIMP (**File → Open**).
2. Choose **Filters → DepthForge → Generate Depth Map…**
3. Pick a mode:
   * **Visual mode** — a good-looking depth map for viewing on screen.
   * **Tactile mode** — for 3D prints read by touch. The result is deliberately
     the *opposite* of a good-looking depth map: fine texture is noise, contours
     must be broad and shallow, and 3–5 discrete height levels read better under
     a fingertip than a smooth gradient. This mode follows museum tyflographic
     guidelines (RNIB, Museo del Prado).
4. The first run is slower (models are loaded into memory). A 2000×1500 px image
   typically takes from under a minute to a few minutes, depending on your CPU.

> **Note:** while it computes, GIMP will look frozen — the window stops
> responding because the work happens in a separate process. That is expected.
> Let it finish.

---

## 4. Troubleshooting

### The plugin isn't in the Filters menu

* Did you **restart** GIMP after installing?
* Your GIMP may use a different config directory. Point the installer at it:

  ```bash
  # Linux
  ./install.sh --gimp-dir ~/.config/GIMP/3.2
  ```
  ```bat
  rem Windows (from a command prompt, in the package folder)
  install.bat --gimp-dir "%APPDATA%\GIMP\3.2"
  ```

  You can find your directory in GIMP under
  **Edit → Preferences → Folders → Plug-Ins**.

### The plugin is there but reports an error

Run the built-in diagnostics: **Filters → DepthForge → Diagnose…**
It reports whether the plugin can see the bundled Python and the app directory.

A detailed log is written to `depthforge_gimp.log` in this package's `app/`
subfolder (or in your system temp directory).

### "The depth map is flat and boring"

That almost always means **the models are missing** — DepthForge silently falls
back to a synthetic estimator (inverted luminance + blur) that produces poor
results. Fetch the models:

```bash
# Linux
./python/bin/python3 app/download_models.py
```
```bat
rem Windows
python\python.exe app\download_models.py
```

### I moved the folder and the plugin broke

Run `install.sh` / `install.bat` again from the new location. The installer
rewrites the stored paths.

### Model download failed

You can re-run the installer at any time — already-downloaded files are not
fetched twice. If your network blocks the download, get the four files manually
from <https://github.com/GrzegorzOle/DepthForge/releases> and place them here:

```
app/models/dpt/openvino/dpt_large.bin
app/models/dpt/openvino/dpt_large.xml
app/models/midas/openvino/midas_v21_small_256.bin
app/models/midas/openvino/midas_v21_small_256.xml
```

---

## 5. Uninstalling

* **Windows** — double-click `uninstall.bat`
* **Linux** — `./uninstall.sh`

This removes the plugin from GIMP. The package folder itself stays — delete it
manually to reclaim the disk space.

---

## 6. What's inside

```
DepthForge-0.1.0-<platform>/
├── install.sh / install.bat        ← the one thing you run
├── uninstall.sh / uninstall.bat
├── python/                          Python 3.12 + numpy, OpenCV, OpenVINO, SciPy
├── app/                             DepthForge code + models (after install)
├── plugin/depthforge/               the plugin, copied into GIMP
├── INSTALL_EN.md                    this file
└── INSTALL_PL.md                    Polish version
```

The package installs nothing system-wide beyond a single plugin folder in your
GIMP config directory. It does not touch the registry, your Python, or GIMP's
Python.

License: see `app/LICENSE`.
Source code and bug reports: <https://github.com/GrzegorzOle/DepthForge>
