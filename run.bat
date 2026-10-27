@echo off
rem Double-click for the menu, or: run.bat detect helmet m quick
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if "%~1"=="" pause
