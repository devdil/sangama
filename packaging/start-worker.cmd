@echo off
setlocal
set /p "CONFIG=Path to your Sangama worker JSON configuration: "
if not defined CONFIG exit /b 1
"%~dp0sangama.exe" mesh --config "%CONFIG%"
pause
