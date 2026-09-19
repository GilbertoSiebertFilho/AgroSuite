# AgroSuite

Local app for agricultural monitor data: trial design, import, cleaning,
terrain, economics and prescriptions. `README.md` describes what it does and
how the tabs relate; this file is only how to work on it here.

## Running

```
run.bat                                  # the launcher: builds .venv, then starts
.venv\Scripts\python.exe -m agrosuite    # same thing once .venv exists
.venv\Scripts\python.exe -m agrosuite --reload --no-browser
```

The server listens on 127.0.0.1:8765, or the first free port above it.

## Tests

```
.venv\Scripts\python.exe -m pytest tests/ -q
```

905 tests, about six minutes.

**Always through `.venv`, never the machine's global Python.** The global
interpreter carries affine 2.4, where `Affine @ (col, row)` does not exist, and
53 terrain/raster tests fail with a `TypeError` at `formats/raster.py:117` that
looks like a defect in the project and is not. `run.bat` installs affine 3.

GitHub Actions runs the same suite on Windows and Ubuntu for every pull
request and every push to `main` (`.github/workflows/tests.yml`). Work goes on
a branch, through a pull request, and into `main` once the check is green.

## Layout

`core/` data model, units, CRS, AB lines · `formats/` read/write per format,
ISOXML, packages, USB, QGIS, SoMat `.sie` · `clean/` filters and report ·
`terrain/` elevation grid, hydrology, contours · `machine/` a telemetry log's
day, fuel, DEF and tyre trampling (the Machine tab) · `difm/` response,
economics, trial layout (the Economics tab) · `app/` local server and
interface · `mcp_server.py` the tool surface Claude drives.

A real SoMat log to try things against lives outside the repo, in
`GIT\CNH_Augmenta\2026\Field_South_Wheat\2_Analisys\01_input\somat\`; the tests
use `machine/synthetic.py`, which writes a day with known answers.

`.mcp.json` registers that MCP server for sessions opened in this folder,
by a path relative to the project (`${CLAUDE_PROJECT_DIR:-.}`) in the
Windows venv layout, `.venv\Scripts\python.exe`.
