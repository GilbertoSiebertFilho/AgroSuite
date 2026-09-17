@echo off
REM ---------------------------------------------------------------------
REM  AgroSuite - Windows launcher
REM
REM  The first run builds a virtual environment and installs what the app
REM  needs. Later runs just start it, and reinstall only when the list of
REM  dependencies has changed since the last install.
REM
REM  That last part is what makes "git pull" safe. A release that needs a
REM  library the previous one did not would otherwise start and then fail
REM  on the first import, with a traceback that says nothing about the
REM  real cause.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set VENV=.venv
set PY=%VENV%\Scripts\python.exe
REM  A copy of the requirements as they were when they were last installed.
REM  Comparing against it is how this script knows an update is needed.
set STAMP=%VENV%\requirements.installed

if not exist "%PY%" (
    echo.
    echo  First run: setting up the environment. This takes a few minutes.
    echo.
    REM  "if errorlevel N" is read when the line runs; "%%errorlevel%%" would
    REM  be read when the block is parsed, before "where" has answered.
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV%"
    ) else (
        python -m venv "%VENV%"
    )
    if errorlevel 1 (
        echo.
        echo  Could not create the virtual environment.
        echo  Install Python 3.10 or newer from https://www.python.org/downloads/
        echo  and tick "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    "%PY%" -m pip install --upgrade pip --quiet
    call :install
    if errorlevel 1 exit /b 1
) else (
    REM  No stamp, or a different one, means the list moved: install again.
    fc /b requirements.txt "%STAMP%" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo  The app needs different libraries than last time. Updating.
        echo.
        call :install
        if errorlevel 1 exit /b 1
    )
)

"%PY%" -m agrosuite %*
if errorlevel 1 pause
endlocal
exit /b 0

:install
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  Failed to install the dependencies. Check your connection and try again.
    echo.
    pause
    exit /b 1
)
REM  Stamped only after a successful install, so an interrupted one is
REM  tried again next time rather than being remembered as done.
copy /y requirements.txt "%STAMP%" >nul
exit /b 0
