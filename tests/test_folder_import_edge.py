"""A lone .shp that is the whole drop.

The folder walk treats a folder holding one .shp as one shapefile. Without
its .shx and .dbf that folder is not a source, and reading it would fail
inside the reader with a temporary path the user never saw. The .shp has to
fall through to the item walk, which names it and says what is missing —
the same answer the user gets when the .shp arrives beside other files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as fx  # noqa: E402

from agrosuite.app import server as server_mod  # noqa: E402
from agrosuite.app import session as session_mod  # noqa: E402

ENDPOINT = "/api/import/files"


@pytest.fixture
def client(monkeypatch):
    fresh = session_mod.Session()
    monkeypatch.setattr(server_mod, "state", fresh)
    with TestClient(server_mod.app) as client:
        yield client
    fresh.cleanup()


@pytest.mark.parametrize("relative", ["JD_Colheita_Canola.shp", "drop/JD_Colheita_Canola.shp"])
def test_a_lone_shp_alone_in_the_drop_is_named_with_what_is_missing(client, tmp_path, relative):
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    response = client.post(
        ENDPOINT, files=[("files", (shp.name, shp.read_bytes()))], data={"paths": [relative]})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "Nothing in the drop could be imported" in detail
    assert f"{relative}: '{shp.name}' arrived without .shx and .dbf" in detail
    assert str(tmp_path) not in detail and "Unable to open" not in detail
    assert server_mod.state.list() == []


def test_a_complete_set_alone_in_a_folder_is_still_one_dataset(client, tmp_path):
    """The fix must not touch the case the folder rule exists for."""
    shp = fx.john_deere_shapefile(tmp_path / "jd")
    members = [p for p in shp.parent.iterdir() if p.stem == shp.stem]
    response = client.post(
        ENDPOINT,
        files=[("files", (p.name, p.read_bytes())) for p in members],
        data={"paths": [f"Field 12/{p.name}" for p in members]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["label"] for item in body["imported"]] == [shp.stem]
    assert body["skipped"] == []
