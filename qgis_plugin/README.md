# AgroSuite plugin for QGIS

Bridges QGIS and AgroSuite running on the same machine. It does not
reimplement anything: AgroSuite already runs as a local server, so the plugin
hands layers across and brings results back.

## Installing

1. Copy the `agrosuite` folder into your QGIS plugins directory:
   - **Windows:** `C:\Users\<you>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS.
3. Enable it under **Plugins → Manage and Install Plugins → Installed**.

## Using it

Start AgroSuite first (`run.bat`, or `python -m agrosuite`). The plugin looks
for it on ports 8765 to 8774 and says so plainly when it is not running.

Three toolbar buttons:

- **Send the active layer to AgroSuite** — writes the selected layer to a
  temporary GeoPackage and loads it, with the preliminary analysis reported
  back in a dialog.
- **Bring a dataset back from AgroSuite** — lists what AgroSuite holds and adds
  the chosen one to the QGIS project.
- **Open AgroSuite** — the browser window, for everything the plugin does not
  wrap.

## Without the plugin

The same round trip works by hand and needs no installation:

- **QGIS → AgroSuite:** save the layer as GeoPackage or shapefile, then use
  *Open by path / folder* in AgroSuite.
- **AgroSuite → QGIS:** the Export tab writes a GeoPackage and a `.qgs`
  project. Open the `.qgs` to get every layer at once, or add the `.gpkg`
  directly.
