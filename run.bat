@echo off
rem Double-click for the menu, or: run.bat detect helmet m quick
cd /d "%~dp0"
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if "%~1"=="" pause
