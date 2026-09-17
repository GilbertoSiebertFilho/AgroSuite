"""Keeping the suite out of the machine it runs on.

The app keeps two things outside the session: the machine profiles and the
recent list under ``AGROSUITE_HOME``, and the projects themselves under the
folder named in its settings — by default ``Documents/AgroSuite``, which is
a folder people keep things in. Neither belongs to a test run, so the whole
run gets a home of its own.

Auto-save is switched off, for a second reason. Several tests start the real
app with ``python -m agrosuite``, and the real app picks the last project up
when it starts: one module's leftovers would arrive in the next module's
empty session, and the failure would read as anything but that. It is
switched off twice over — in the settings file here, and through
``AGROSUITE_AUTOSAVE``, which is the switch that keeps the writer out of a
process altogether. The second one is what covers the tests that point
``AGROSUITE_HOME`` at a home of their own and so never see this settings
file; they pass the environment on to the app they launch.

The tests that are about auto-save (``test_autosave.py``) switch it back on
for themselves, which is also the only way to be sure the switch means
something.

This runs at import, before any fixture and before any subprocess: the app
reads the environment on every call, and a test that launches the app hands
it this same environment.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_HOME = Path(tempfile.mkdtemp(prefix="agrosuite-tests-"))
os.environ["AGROSUITE_HOME"] = str(_HOME)
os.environ["AGROSUITE_AUTOSAVE"] = "off"
(_HOME / "settings.json").write_text(
    json.dumps({"version": 1, "projects_dir": str(_HOME / "projects"), "autosave": False},
               indent=2) + "\n",
    encoding="utf-8",
)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest's signature
    shutil.rmtree(_HOME, ignore_errors=True)
