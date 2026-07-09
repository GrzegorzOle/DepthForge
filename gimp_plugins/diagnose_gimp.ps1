# DepthForge – Narzędzie diagnostyczne GIMP
# Uruchom w PowerShell: .\gimp_plugins\diagnose_gimp.ps1

$PLUGIN_DIR  = "$env:APPDATA\GIMP\3.0\plug-ins\depthforge"
$LOG_PLUGIN  = "$PLUGIN_DIR\depthforge_run.log"
$LOG_TEMP    = "$env:TEMP\depthforge_run.log"
$GIMP_EXE    = "C:\Users\grzeg\AppData\Local\Programs\GIMP 3\bin\gimp.exe"
$GIMP_CACHE  = "$env:APPDATA\GIMP\3.0"

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
    Write-Host "      Uruchom: python H:\test\DepthForge\install.py --gimp-only --yes"
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
Write-Host ""
Write-Host "[ 3 ] Zmienna DEPTHFORGE_PYTHON:" -ForegroundColor Yellow
$env_val = [System.Environment]::GetEnvironmentVariable("DEPTHFORGE_PYTHON", "User")
if ($env_val) {
    Write-Host "      $env_val" -ForegroundColor Green
    if (Test-Path $env_val) {
        Write-Host "      Plik istnieje: OK" -ForegroundColor Green
    } else {
        Write-Host "      UWAGA: plik nie istnieje!" -ForegroundColor Red
    }
} else {
    Write-Host "      (nie ustawiona – wtyczka uzyje depthforge_install.json)" -ForegroundColor Yellow
}

# 4. Test zewnetrznego Pythona
Write-Host ""
Write-Host "[ 4 ] Test zewnetrznego Pythona (numpy + opencv):" -ForegroundColor Yellow
$py = if ($env_val -and (Test-Path $env_val)) { $env_val } elseif ($cfg -and (Test-Path $cfg.venv_python)) { $cfg.venv_python } else { $null }
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
if (Test-Path $GIMP_EXE) {
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
    $py_sys = Get-Command python -ErrorAction SilentlyContinue
    if ($py_sys) {
        $syntax = & python -c "import ast; ast.parse(open(r'$PLUGIN_DIR\depthforge.py', encoding='utf-8').read()); print('Syntax OK')" 2>&1
        Write-Host "  Sprawdzenie skladni: $syntax" -ForegroundColor $(if ($syntax -like "*OK*") {"Green"} else {"Red"})
    }

    Write-Host ""
    Write-Host "  b) Uruchom GIMP z konsoli zeby zobaczyc bledy:" -ForegroundColor Yellow
    Write-Host "     & '$GIMP_EXE' 2>&1 | Tee-Object gimp_errors.txt" -ForegroundColor Cyan
}

Write-Host ""
Read-Host "Nacisnij Enter aby zakonczyc"

