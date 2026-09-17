#!/usr/bin/env bash
# AgroSuite - launcher for Linux and macOS.
#
# The first run builds the virtual environment; later runs start the app and
# reinstall only when the dependency list has changed since the last install,
# so pulling a release that needs a new library cannot leave the app failing
# on its first import.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
# A copy of the requirements as last installed; comparing against it is how
# this script knows an update is needed.
STAMP="$VENV/requirements.installed"

install() {
  "$PY" -m pip install -r requirements.txt
  # Stamped only after a successful install, so an interrupted one is tried
  # again next time rather than being remembered as done.
  cp requirements.txt "$STAMP"
}

if [ ! -x "$PY" ]; then
  echo
  echo "  First run: setting up the environment."
  echo
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip --quiet
  install
elif ! cmp -s requirements.txt "$STAMP"; then
  echo
  echo "  The app needs different libraries than last time. Updating."
  echo
  install
fi

exec "$PY" -m agrosuite "$@"
