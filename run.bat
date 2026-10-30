@echo off
rem Double-click for the menu, or: run train D:\data\helmet   (run setup installs `run` for every terminal)
setlocal
set "MLT_CWD=%CD%"
cd /d "%~dp0"
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" if not defined MLT_NOPAUSE pause
exit /b %RC%
