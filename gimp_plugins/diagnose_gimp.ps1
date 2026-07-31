# DepthForge – Narzędzie diagnostyczne GIMP
# Uruchom w PowerShell: .\gimp_plugins\diagnose_gimp.ps1

# Nic tu nie moze byc zaszyte na sztywno: skrypt ma dzialac na maszynie
# uzytkownika, a nie tylko na tej, na ktorej powstal. Katalog konfiguracyjny
# GIMP-a wybieramy najnowszy z istniejacych (3.0, 3.2, ...), bo stara instalacja
# potrafi zostac na dysku i myli diagnostyke.
$GIMP_BASE = "$env:APPDATA\GIMP"
$GIMP_VER  = if (Test-Path $GIMP_BASE) {
    Get-ChildItem $GIMP_BASE -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d+\.\d+$' } |
        Sort-Object { [version]$_.Name } -Descending |
        Select-Object -First 1 -ExpandProperty Name
} else { $null }
if (-not $GIMP_VER) { $GIMP_VER = "3.2" }

$GIMP_CACHE  = "$GIMP_BASE\$GIMP_VER"
$PLUGIN_DIR  = "$GIMP_CACHE\plug-ins\depthforge"
$LOG_PLUGIN  = "$PLUGIN_DIR\depthforge_run.log"
$LOG_TEMP    = "$env:TEMP\depthforge_run.log"

$GIMP_EXE = @(
    "$env:LOCALAPPDATA\Programs\GIMP 3\bin\gimp.exe",
    "$env:ProgramFiles\GIMP 3\bin\gimp.exe",
    "${env:ProgramFiles(x86)}\GIMP 3\bin\gimp.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $GIMP_EXE) {
    $cmd = Get-Command gimp.exe, gimp-3.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($cmd) { $GIMP_EXE = $cmd.Source }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  DepthForge – Diagnostyka GIMP" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Sprawdz pliki wtyczki
Write-Host "[ 1 ] Pliki wtyczki:" -ForegroundColor Yellow
if (Test-Path $PLUGIN_DIR) {
    Get-ChildItem $PLUGIN_DIR | ForEach-Object {
        Write-Host "      $($_.Name)  ($($_.Length) B)  $($_.LastWriteTime)" -ForegroundColor Green
    }
} else {
    Write-Host "      BRAK folderu: $PLUGIN_DIR" -ForegroundColor Red
    Write-Host "      Uruchom ponownie instalator DepthForge (setup.exe lub install.py)"
    exit 1
}

# 2. Sprawdz install JSON
Write-Host ""
Write-Host "[ 2 ] Konfiguracja (depthforge_install.json):" -ForegroundColor Yellow
$json_path = "$PLUGIN_DIR\depthforge_install.json"
if (Test-Path $json_path) {
    $cfg = Get-Content $json_path | ConvertFrom-Json
    Write-Host "      project_root : $($cfg.project_root)" -ForegroundColor Green
    Write-Host "      venv_python  : $($cfg.venv_python)" -ForegroundColor Green
    if (Test-Path $cfg.venv_python) {
        Write-Host "      venv Python  : OK (plik istnieje)" -ForegroundColor Green
    } else {
        Write-Host "      venv Python  : BRAK pliku $($cfg.venv_python)" -ForegroundColor Red
    }
} else {
    Write-Host "      BRAK pliku install JSON" -ForegroundColor Red
}

# 3. Sprawdz env var
# Kolejnosc musi byc taka sama jak w _find_external_python(): najpierw install
# JSON, potem zmienna srodowiskowa. Inaczej ten raport pokazuje inny interpreter
# niz ten, ktorego naprawde uzywa wtyczka - a to wlasnie taka rozbieznosc
# kosztowala cala sesje debugowania.
Write-Host ""
Write-Host "[ 3 ] Zmienna DEPTHFORGE_PYTHON:" -ForegroundColor Yellow
$env_val = [System.Environment]::GetEnvironmentVariable("DEPTHFORGE_PYTHON", "User")
$json_py = if ($cfg -and $cfg.venv_python -and (Test-Path $cfg.venv_python)) { $cfg.venv_python } else { $null }
if ($env_val) {
    Write-Host "      $env_val"
    if (-not (Test-Path $env_val)) {
        Write-Host "      UWAGA: plik nie istnieje!" -ForegroundColor Red
    }
    if ($json_py) {
        if ((Resolve-Path $json_py).Path -ieq (Resolve-Path $env_val -ErrorAction SilentlyContinue).Path) {
            Write-Host "      Zgodna z install JSON: OK" -ForegroundColor Green
        } else {
            Write-Host "      IGNOROWANA - install JSON ma pierwszenstwo." -ForegroundColor Yellow
            Write-Host "      Prawdopodobnie pozostalosc po starszej instalacji." -ForegroundColor Yellow
            Write-Host "      Usun ja: [Environment]::SetEnvironmentVariable('DEPTHFORGE_PYTHON',`$null,'User')" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "      (nie ustawiona – wtyczka uzyje depthforge_install.json)" -ForegroundColor Yellow
}

# 4. Test zewnetrznego Pythona
Write-Host ""
Write-Host "[ 4 ] Test zewnetrznego Pythona (numpy + opencv):" -ForegroundColor Yellow
$py = if ($json_py) { $json_py } elseif ($env_val -and (Test-Path $env_val)) { $env_val } else { $null }
if ($py) {
    $result = & $py -c "import numpy, cv2; print(f'numpy {numpy.__version__}, opencv {cv2.__version__}')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      OK: $result" -ForegroundColor Green
    } else {
        Write-Host "      BLAD: $result" -ForegroundColor Red
    }
} else {
    Write-Host "      Brak interpretera Python do testu" -ForegroundColor Red
}

# 5. Stare logi
Write-Host ""
Write-Host "[ 5 ] Logi z poprzednich uruchomien:" -ForegroundColor Yellow
foreach ($log in @($LOG_PLUGIN, $LOG_TEMP)) {
    if (Test-Path $log) {
        Write-Host "      LOG: $log" -ForegroundColor Green
        Get-Content $log | Select-Object -Last 20 | ForEach-Object { Write-Host "        $_" }
    } else {
        Write-Host "      Brak logu: $log" -ForegroundColor Gray
    }
}

# 6. Wyczysc stare logi i cache GIMP
Write-Host ""
Write-Host "[ 6 ] Czyszczenie cache GIMP i starych logow..." -ForegroundColor Yellow

# Usun stare logi
foreach ($log in @($LOG_PLUGIN, $LOG_TEMP)) {
    if (Test-Path $log) { Remove-Item $log -Force; Write-Host "      Usunieto log: $log" }
}

# Wyczysc cache procedur GIMP (wymusza ponowne zaladowanie wtyczek)
$cache_files = @(
    "$GIMP_CACHE\pluginrc",
    "$GIMP_CACHE\pluginrc.bak"
)
foreach ($f in $cache_files) {
    if (Test-Path $f) {
        Remove-Item $f -Force
        Write-Host "      Usunieto cache: $f" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Uruchamiam GIMP..." -ForegroundColor Cyan
Write-Host "  Po zamknieciu GIMP sprawdze logi." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  INSTRUKCJA w GIMP:" -ForegroundColor White
Write-Host "  1. Otworz dowolne zdjecie (Plik -> Otworz)" -ForegroundColor White
Write-Host "  2. Kliknij: Filtry -> DepthForge -> Generate Depth Map..." -ForegroundColor White
Write-Host "  3. Kliknij OK" -ForegroundColor White
Write-Host "  4. Zamknij GIMP" -ForegroundColor White
Write-Host ""

# Uruchom GIMP i czekaj na zamkniecie
if ($GIMP_EXE -and (Test-Path $GIMP_EXE)) {
    Start-Process $GIMP_EXE -Wait
} else {
    # Sprobuj znalezc GIMP w innych lokalizacjach
    $gimp_alt = Get-ChildItem "$env:LOCALAPPDATA\Programs\GIMP*\bin\gimp*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($gimp_alt) {
        Start-Process $gimp_alt.FullName -Wait
    } else {
        Write-Host "  Nie mozna znalezc GIMP.exe – uruchom GIMP recznie, zamknij, a potem ponow skrypt." -ForegroundColor Yellow
        Read-Host "  Nacisnij Enter gdy GIMP zostanie zamkniety"
    }
}

# Po zamknieciu GIMP – pokaz logi
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GIMP zamkniety. Wyniki diagnostyki:" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$found_log = $false
foreach ($log in @($LOG_PLUGIN, $LOG_TEMP)) {
    if (Test-Path $log) {
        $found_log = $true
        Write-Host "LOG: $log" -ForegroundColor Green
        Write-Host "---" -ForegroundColor Gray
        Get-Content $log | ForEach-Object { Write-Host "  $_" }
        Write-Host ""
    }
}

if (-not $found_log) {
    Write-Host "  BRAK logow – wtyczka nie zaladowala sie!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Mozliwe przyczyny:" -ForegroundColor Yellow
    Write-Host "  a) Blad skladni w pliku depthforge.py – sprawdz ponizej:" -ForegroundColor Yellow
    Write-Host ""

    # Przetestuj plik Python
    # Uzyj interpretera z instalacji; systemowego Pythona na maszynie klienta
    # zwyczajnie nie ma i nie musi byc.
    $py_chk = if ($py) { $py } else {
        $c = Get-Command python -ErrorAction SilentlyContinue
        if ($c) { $c.Source } else { $null }
    }
    if ($py_chk) {
        $syntax = & $py_chk -c "import ast; ast.parse(open(r'$PLUGIN_DIR\depthforge.py', encoding='utf-8').read()); print('Syntax OK')" 2>&1
        Write-Host "  Sprawdzenie skladni: $syntax" -ForegroundColor $(if ($syntax -like "*OK*") {"Green"} else {"Red"})
    }

    if ($GIMP_EXE) {
        Write-Host ""
        Write-Host "  b) Uruchom GIMP z konsoli zeby zobaczyc bledy:" -ForegroundColor Yellow
        Write-Host "     & '$GIMP_EXE' 2>&1 | Tee-Object gimp_errors.txt" -ForegroundColor Cyan
    }
}

Write-Host ""
Read-Host "Nacisnij Enter aby zakonczyc"

