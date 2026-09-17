"""AgroSuite inside QGIS.

The plugin does not reimplement anything: AgroSuite already runs as a local
server, so the plugin is a thin bridge that hands layers across and brings
results back. Three actions cover the work:

* **Send the active layer** — writes it to a GeoPackage in a temporary folder
  and tells AgroSuite where it is. Nothing travels over the network beyond
  localhost, and nothing is copied that QGIS could not already see.
* **Bring a dataset back** — lists what AgroSuite currently holds and adds the
  chosen one to the QGIS project.
* **Open AgroSuite** — the browser window, for everything the plugin does not
  wrap.

The server has to be running. The plugin says so plainly instead of failing
with a connection error, because "AgroSuite is not running" is a thing the
user can act on and a stack trace is not.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
import webbrowser

from qgis.core import QgsProject, QgsVectorFileWriter, QgsVectorLayer
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QInputDialog, QMessageBox

#: Where AgroSuite listens. It picks the next free port if 8765 is taken, so
#: the plugin tries a small range rather than giving up on the first refusal.
PORTS = range(8765, 8775)
TIMEOUT_S = 30


class AgroSuitePlugin:
    """Bridge between the QGIS project and the AgroSuite server."""

    def __init__(self, iface):
        self.iface = iface
        self.actions: list[QAction] = []
        self.menu = "AgroSuite"
        self._base_url: str | None = None

    # -- QGIS lifecycle --------------------------------------------------
    def initGui(self):  # noqa: N802 - the name QGIS requires
        for text, handler, tip in (
            ("Send the active layer to AgroSuite", self.send_active_layer,
             "Write the selected layer to a GeoPackage and load it in AgroSuite."),
            ("Bring a dataset back from AgroSuite", self.fetch_dataset,
             "List what AgroSuite holds and add one of them to this project."),
            ("Open AgroSuite", self.open_app,
             "Open the AgroSuite window in the browser."),
        ):
            action = QAction(self.tr(text), self.iface.mainWindow())
            action.setStatusTip(self.tr(tip))
            action.triggered.connect(handler)
            self.iface.addPluginToVectorMenu(self.menu, action)
            self.iface.addToolBarIcon(action)
            self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginVectorMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions.clear()

    def tr(self, message: str) -> str:
        return QCoreApplication.translate("AgroSuite", message)

    # -- talking to the server -------------------------------------------
    def base_url(self) -> str | None:
        """Find the running AgroSuite, remembering where it was."""
        if self._base_url and self._probe(self._base_url):
            return self._base_url
        for port in PORTS:
            url = f"http://127.0.0.1:{port}"
            if self._probe(url):
                self._base_url = url
                return url
        return None

    @staticmethod
    def _probe(base: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                return json.load(response).get("ok") is True
        except Exception:
            return False

    def call(self, path: str, body=None):
        base = self.base_url()
        if base is None:
            self.warn(
                "AgroSuite is not running",
                "Start AgroSuite (run.bat, or 'python -m agrosuite') and try again. "
                "The plugin looks for it on ports 8765 to 8774.",
            )
            return None
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            base + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail", str(error))
            except Exception:
                detail = str(error)
            self.warn("AgroSuite refused the request", detail)
        except Exception as error:
            self.warn("Could not reach AgroSuite", str(error))
        return None

    # -- actions ----------------------------------------------------------
    def send_active_layer(self):
        layer = self.iface.activeLayer()
        if not isinstance(layer, QgsVectorLayer):
            self.warn("No vector layer selected",
                      "Select a vector layer in the Layers panel first.")
            return

        folder = tempfile.mkdtemp(prefix="qgis_to_agrosuite_")
        target = os.path.join(folder, f"{_safe(layer.name())}.gpkg")

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = _safe(layer.name())
        # AgroSuite reprojects on import, so the layer goes across as it is.
        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, target, QgsProject.instance().transformContext(), options
        )
        # The return shape differs across QGIS versions; only the first element
        # is the error code in all of them.
        code = error[0] if isinstance(error, tuple) else error
        if code != QgsVectorFileWriter.NoError:
            self.warn("Could not write the layer", f"QGIS returned error {code}.")
            return

        result = self.call("/api/import/path", {"path": target})
        if result is None:
            return
        preflight = result.get("preflight") or {}
        self.info(
            "Sent to AgroSuite",
            f"{result['label']}: {result['rows']} records, read as "
            f"{result['meta']['operation_label']}.\n\n"
            + (preflight.get("summary") or ""),
        )

    def fetch_dataset(self):
        listing = self.call("/api/datasets")
        if listing is None:
            return
        datasets = listing.get("datasets", [])
        if not datasets:
            self.info("Nothing loaded", "AgroSuite is not holding any dataset yet.")
            return

        labels = [f"{d['label']} ({d['rows']} records)" for d in datasets]
        choice, accepted = QInputDialog.getItem(
            self.iface.mainWindow(), "Bring a dataset back",
            "Which dataset?", labels, 0, False,
        )
        if not accepted:
            return
        dataset = datasets[labels.index(choice)]

        folder = tempfile.mkdtemp(prefix="agrosuite_to_qgis_")
        result = self.call("/api/export", {
            "dataset_id": dataset["id"],
            "formats": ["geojson"],
            "task_name": _safe(dataset["label"]),
        })
        if result is None:
            return

        path = next(
            (output.get("path") for output in result.get("outputs", [])
             if output.get("format") == "geojson"), None,
        )
        if not path or not os.path.exists(path):
            self.warn("The file did not arrive",
                      "AgroSuite reported success but the file is not where it said.")
            return

        added = QgsVectorLayer(path, dataset["label"], "ogr")
        if not added.isValid():
            self.warn("QGIS could not read the layer", path)
            return
        QgsProject.instance().addMapLayer(added)
        self.info("Added to the project", f"{dataset['label']} is now in the Layers panel.")

    def open_app(self):
        base = self.base_url()
        if base is None:
            self.warn(
                "AgroSuite is not running",
                "Start it with run.bat, or 'python -m agrosuite' from the project folder.",
            )
            return
        webbrowser.open(base)

    # -- dialogs ----------------------------------------------------------
    def info(self, title: str, message: str):
        QMessageBox.information(self.iface.mainWindow(), title, message)

    def warn(self, title: str, message: str):
        QMessageBox.warning(self.iface.mainWindow(), title, message)


def _safe(name: str) -> str:
    """A layer name safe for a file path and a GeoPackage table."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:60] or "layer"
