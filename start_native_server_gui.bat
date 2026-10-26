@echo off
setlocal
set "PYTHON=F:\Models\infernix-env\python.exe"
set "PYTHONW=F:\Models\infernix-env\pythonw.exe"
set "APP=%~dp0tools\native_server_gui.py"
if not exist "%APP%" (
  echo Missing GUI: "%APP%"
  pause
  exit /b 1
)
if exist "%PYTHONW%" (
  start "" "%PYTHONW%" "%APP%" %*
  exit /b 0
)
if exist "%PYTHON%" (
  start "Qwen experimental server GUI" "%PYTHON%" "%APP%" %*
  exit /b 0
)
echo Python environment was not found at F:\Models\infernix-env.
pause
exit /b 1