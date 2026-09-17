"""The folder-drop controls in the interface, checked against the contract.

The interface is vanilla JavaScript and has no unit tests of its own; what
can be checked here without a browser is the contract it relies on. The
request the page builds must carry the field names the route reads, in the
order the route pairs them; the controls it binds must exist in the page;
the entry API must be called before the drop handler first yields (the item
list is gone after that); and the cap must be applied before the upload
starts. Each of these would otherwise only show up in a browser, on a
real folder.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app.routes import folder_import  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def test_controls_are_in_the_page():
    for node_id in ("btn-open-file", "btn-open-folder", "file-input", "folder-input", "drop-report"):
        assert f'id="{node_id}"' in INDEX, f"index.html has no #{node_id}"
    panel = INDEX[INDEX.index("<h3>Load data</h3>"):INDEX.index("<h3>Project</h3>")]
    assert 'id="btn-open-folder"' in panel, "the folder button belongs beside the file button"
    assert 'id="drop-report"' in panel, "the skipped list belongs under Load data"
    # A shapefile's four parts are picked together; a folder comes with its tree.
    assert re.search(r'<input[^>]*id="file-input"[^>]*\bmultiple\b', INDEX)
    assert re.search(r'<input[^>]*id="folder-input"[^>]*\bwebkitdirectory\b', INDEX)


def test_the_page_posts_what_the_route_reads():
    served = {route.path for route in folder_import.router.routes}
    assert "/api/import/files" in served
    block = _block("importFiles")
    assert '"/api/import/files"' in block, "the page never calls the tree import"
    # One 'files' part and one 'paths' field per entry, in the same order:
    # the route pairs them by position. It reads the form itself, so the
    # field names live in its source rather than in its signature.
    route = inspect.getsource(folder_import.import_files)
    assert 'form.getlist("files")' in route and 'form.getlist("paths")' in route
    files_at = block.index('form.append("files"')
    paths_at = block.index('form.append("paths"')
    assert files_at < paths_at
    loop = block[block.rindex("for (", 0, files_at):paths_at]
    assert "form.append(\"files\"" in loop, "files and paths must be appended in one loop"
    # The body is a FormData, and api() must pass one through untouched.
    assert "new FormData()" in block
    assert "options.body instanceof FormData ? options.body" in _block("api")


def test_both_pickers_and_the_drop_zone_reach_the_tree_import():
    imports = _block("bindImport")
    assert '"folder-input"' in imports and '"btn-open-folder"' in imports
    assert "webkitRelativePath" in imports, "the folder picker's paths come from webkitRelativePath"
    assert imports.count("this.importFiles(") == 2, "pickers and drop zone"
    assert "this.collectDropped(e.dataTransfer)" in imports
    # A project file still replaces the session instead of being imported.
    assert imports.count("this.isProjectFile(file.name)") == 2


def test_entries_are_taken_before_the_first_await():
    """DataTransferItemList empties once the drop handler yields; every
    webkitGetAsEntry call has to come first, and the walk after."""
    block = _block("collectDropped")
    assert block.index("webkitGetAsEntry") < block.index("await ")
    assert "fullPath" in block, "paths come from the entry, not the file name"
    assert "transfer?.files" in block, "no entry API means plain files, not nothing"
    assert "readEntries" in block and "for (;;)" in block, "a folder is read until an empty batch"


def test_the_cap_and_the_busy_state():
    block = _block("importFiles")
    cap_at = block.index("maxFiles")
    assert "2000" in block and "2 * 1024 ** 3" in block
    # The server parses up to this many files; a drop the page lets through
    # must not be uploaded whole and then refused at the parser.
    assert f"maxFiles = {folder_import.MAX_FILES}," in block
    assert cap_at < block.index('"/api/import/files"'), "the cap comes before the upload"
    assert 'this.busy(document.getElementById("app")' in block
    # After a 200: the list, the first dataset selected, the skips shown.
    assert "this.refreshDatasets()" in block
    assert "this.selectDataset(result.imported[0].id)" in block
    assert "this.renderDropReport(result.skipped)" in block
    assert "dataset(s)" in block


def test_skipped_files_are_listed_with_their_reason():
    block = _block("renderDropReport")
    assert 'details class="fold"' in block
    assert "this.escape(item.name)" in block and "this.escape(item.reason)" in block
    assert 'box.innerHTML = ""' in block, "a clean drop clears the previous list"


def test_no_portuguese_slipped_in():
    for path in ("app.js", "index.html", "style.css"):
        text = (STATIC / path).read_text(encoding="utf-8")
        for word in ("arquivo", "pasta", "importar", "ignorad"):
            assert not re.search(rf"\b{word}", text, flags=re.IGNORECASE), f"{word!r} in {path}"
