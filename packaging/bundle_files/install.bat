@echo off
rem DepthForge - installer wrapper.
rem Runs the installer with the CPython interpreter bundled in this folder,
rem so no system Python is required.
setlocal
set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"

if not exist "%PY%" (
    echo ERROR: bundled Python not found at %PY%
    echo The archive looks incomplete - unpack it again.
    pause
    exit /b 1
)

"%PY%" "%HERE%bundle_install.py" %*
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
