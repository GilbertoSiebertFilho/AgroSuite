@echo off
REM ---------------------------------------------------------------------
REM  Puts an AgroSuite shortcut in the Start menu (and on the desktop when
REM  it can), with the app's icon. Double-click it once after copying the
REM  folder to a computer. The work is in create_shortcut.ps1.
REM ---------------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1"
pause
