"""Leftovers from the second review pass, pinned here.

Three of them: the demo strip trial drawn at corn's ten tonnes a hectare while
the app opens on canola, so the first thing a new user saw was the app's own
check calling its own sample data wrong; the package endpoint answering with
``observacoes`` while the page read ``notes``, so the rate-conversion note
never reached the screen; and a handful of Portuguese defaults — ``Talhao``,
``Prescricao``, ``Produto``, ``Fazenda``, ``pacote_``, ``_contorno`` — that
went out in file names and ISOXML designators. The sweep at the end is what
would have caught them: it fails on the next one too.

The reload that left nothing selected is pinned in ``test_regression_fixes_ui``,
which drives it in a browser; it is not repeated here.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402
from agrosuite.core import preflight  # noqa: E402
from agrosuite.demo import synthetic_trial  # noqa: E402
from agrosuite.formats import isoxml  # noqa: E402

STATIC = ROOT / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")

# A quarter section south of Moose Jaw, as (lon, lat).
QUARTER_SECTION = [
    (-105.8340, 50.4520), (-105.8229, 50.4520),
    (-105.8229, 50.4592), (-105.8340, 50.4592),
]


@pytest.fixture
def client(monkeypatch):
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    with TestClient(server_mod.app) as client:
        yield client
    fresh.cleanup()


# ---------------------------------------------------------- the demo trial ---

def test_the_trial_demo_is_a_canola_trial_at_canola_yields():
    """The app opens on canola and bu/ac; a trial at ten tonnes a hectare
    read as 200 bu/ac of canola, and the first look proposed kg/alqueire."""
    ds = synthetic_trial()
    ds.ensure_derived()
    assert ds.meta.crop == "canola"

    median = float(np.median(ds.df["value"]))
    assert 2_000.0 <= median <= 3_500.0, f"median {median:.0f} kg/ha is not a canola yield"

    report = preflight.run(ds)
    titles = {f["title"] for f in report["findings"] if f["level"] != "ok"}
    assert "Units look wrong" not in titles
    assert report["proposed_units"] is None
    assert report["verdict"] == "ok"


# ------------------------------------------------------ what goes on the stick ---

def _designators(folder: Path) -> dict[str, str]:
    """Customer, farm, field, task and product names as the terminal will list them."""
    root = ET.parse(next(folder.rglob("TASKDATA.XML"))).getroot()
    names = {
        "customer": root.find("CTR").get("B"),
        "farm": root.find("FRM").get("B"),
        "field": root.find("PFD").get("C"),
    }
    if root.find("TSK") is not None:
        names["task"] = root.find("TSK").get("B")
    if root.find("PDT") is not None:
        names["product"] = root.find("PDT").get("B")
    return names


def test_isoxml_defaults_read_in_english_on_the_terminal(tmp_path):
    """What the user did not type is what the cab screen shows: it has to be
    the language of the rest of the app, not the one the writer was born in."""
    isoxml.write_field_setup(tmp_path / "setup", boundary=QUARTER_SECTION)
    assert _designators(tmp_path / "setup") == {
        "customer": "AgroSuite", "farm": "Farm", "field": "Field", "task": "Setup Field",
    }

    isoxml.write_prescription(
        tmp_path / "rx", grid=np.array([[100.0, 120.0], [140.0, 160.0]]),
        min_lon=-105.8340, min_lat=50.4520, cell_lon=0.0001, cell_lat=0.0001,
    )
    assert _designators(tmp_path / "rx") == {
        "customer": "AgroSuite", "farm": "Farm", "field": "Field",
        "task": "Prescription", "product": "Product",
    }


def test_the_package_answers_with_the_key_the_page_reads(client):
    """The page renders ``result.notes``; a server answering ``observacoes``
    kept the rate-conversion note off the screen without any error."""
    response = client.post("/api/export/package", json={
        "monitor": "raven",
        "boundary": [list(point) for point in QUARTER_SECTION],
    })
    assert response.status_code == 200, response.text
    body = response.json()

    assert "notes" in body and isinstance(body["notes"], list)
    assert "observacoes" not in body
    assert "result.notes" in APP_JS, "the page reads the notes under this key"

    # The folder and the boundary file are what the user sees in the ZIP.
    assert Path(body["folder"]).name.startswith("package_")
    boundary = [c for c in body["contents"] if c["artifact"] == "boundary"]
    assert boundary and Path(boundary[0]["path"]).stem.endswith("_boundary")
    # The check over the written files still recognises the boundary by name.
    assert body["verification"]["totals"]["fail"] == 0, body["verification"]["summary"]


# -------------------------------------------------------------- the sweep ---

# Defaults are quoted and capitalised; the lower-case aliases the readers use
# to recognise Brazilian file names ("contorno", "produto") stay, on purpose.
LEFTOVERS = re.compile(
    r'"(?:Talhao|Prescricao|Produto|Fazenda)"'
    r"|observacoes|pacote_|_contorno"
    r"|\b(?:não|também|já|então|padrão|talhão|ensaio de)\b"
)

SOURCES = sorted(
    [p for suffix in ("py", "js", "html", "css") for p in (ROOT / "agrosuite").rglob(f"*.{suffix}")]
    + list((ROOT / "docs").glob("*.md"))
    + list((ROOT / "qgis_plugin").glob("*.py"))
    + [ROOT / "README.md"]
)


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_portuguese_default_is_left_in(source):
    """Everything the user can see or take to the monitor is in English."""
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
        if LEFTOVERS.search(line)
    ]
    assert not hits, "\n".join(hits)
