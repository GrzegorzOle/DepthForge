@echo off
rem DepthForge - removes the plugin from GIMP. The bundle folder stays.
setlocal
set "HERE=%~dp0"
"%HERE%python\python.exe" "%HERE%bundle_install.py" --uninstall %*
echo.
pause
