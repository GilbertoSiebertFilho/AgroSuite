"""QGIS plugin entry point."""


def classFactory(iface):  # noqa: N802 - the name QGIS requires
    from .plugin import AgroSuitePlugin

    return AgroSuitePlugin(iface)
