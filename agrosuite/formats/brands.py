"""Identifying which monitor a file came from.

There is no universal header saying "this file came from a John Deere". What
exists are signatures: characteristic column names each export program
produces, folder markers and proprietary extensions. This module gathers
those signatures, scores every manufacturer against the file at hand and
returns the most likely one.

The signatures come from the real exports of each platform; where a
manufacturer can export in several formats — the common case — it appears
with several signature entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schema import normalize_name


@dataclass(frozen=True)
class BrandProfile:
    """Profile of a monitor manufacturer or platform."""

    key: str
    label: str
    #: Columns whose mere presence strongly indicates this manufacturer.
    signature_columns: tuple[str, ...] = ()
    #: Supporting columns: they add points but do not decide on their own.
    hint_columns: tuple[str, ...] = ()
    #: File and folder names typical of the export structure.
    path_markers: tuple[str, ...] = ()
    #: Proprietary extensions associated with the platform.
    extensions: tuple[str, ...] = ()
    #: Units the platform normally exports in.
    default_units: dict[str, str] = field(default_factory=dict)
    #: Formats AgroSuite can **write** for this platform.
    export_formats: tuple[str, ...] = ()
    notes: str = ""


BRANDS: tuple[BrandProfile, ...] = (
    BrandProfile(
        key="john_deere",
        label="John Deere",
        signature_columns=(
            "vryieldvol", "vryieldmass", "yld_vol_dr", "yld_mass_d",
            "jd_machine", "machine_model", "operation_operator",
        ),
        hint_columns=(
            "client", "farm", "field", "task", "product", "variety",
            "crop", "machine", "operator", "recording_status",
        ),
        path_markers=(
            "rcd", "setup", "documentation", "gs3_2630", "gs2_2600",
            "operations_center", "myjohndeere",
        ),
        extensions=(".gsd", ".fdd", ".jdf"),
        default_units={"yield": "bu/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "isoxml", "csv"),
        notes=(
            "Operations Center and SMS exports come out as shapefile or CSV. The "
            "binary .gsd/.fdd files from GreenStar 2/3 are proprietary and have to "
            "be converted in their own software before importing."
        ),
    ),
    BrandProfile(
        key="ag_leader",
        label="Ag Leader / SMS",
        signature_columns=(
            "obj_id", "obj__id", "swth_wdth_", "yld_vol_dr", "yld_mass_w",
            "prod_ac_h", "duration_s", "track_deg_",
        ),
        hint_columns=("distance_f", "elevation_", "time", "crop_flow_v"),
        path_markers=("sms", "agleader", "agdata"),
        extensions=(".agdata", ".ilf"),
        default_units={"yield": "bu/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "csv"),
        notes="SMS Advanced is the most common bridge format between platforms.",
    ),
    BrandProfile(
        key="raven",
        label="Raven / Viper",
        signature_columns=(
            "vipertarget", "viper_rate", "raven_rate", "prod_rate", "boom_section",
            "target_rate_1", "applied_rate_1",
        ),
        hint_columns=("section", "product", "tank_mix", "swath", "coverage"),
        path_markers=("viper", "raven", "slingshot", "vt_files", "cn1"),
        extensions=(".cn1", ".vpr"),
        default_units={"rate": "gal/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "csv"),
        notes=(
            "The Viper 4 reads a prescription from a polygon shapefile with a "
            "numeric rate column chosen on the monitor itself."
        ),
    ),
    BrandProfile(
        key="trimble",
        label="Trimble",
        signature_columns=("agdata_id", "tgt_rate_o", "trimble_rate", "fmx_rate"),
        hint_columns=("coverage", "applied", "material", "swath", "vehicle"),
        path_markers=("aggps", "agdata", "fmx", "gfx", "tmx", "trimble", "farmworks"),
        extensions=(".agdata", ".vrt"),
        default_units={"rate": "lb/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "isoxml", "csv"),
        notes="FmX/GFX/TMX accept a shapefile Rx; ISOBUS models also read ISOXML.",
    ),
    BrandProfile(
        key="case_ih",
        label="Case IH (AFS)",
        signature_columns=("afs_rate", "afs_field", "voyager_id"),
        hint_columns=("grower", "farm", "field", "load", "operator", "task"),
        path_markers=("afs", "case", "voyager", "cnh", "advanced_farming",
                      "case_ih", "cnh_industrial", "afs_pro", "afs_connect"),
        extensions=(".vy1", ".vy2", ".vy3"),
        default_units={"yield": "bu/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "isoxml", "csv"),
        notes="AFS Pro 700 and AFS Vision read ISOXML; Voyager formats are proprietary.",
    ),
    BrandProfile(
        key="new_holland",
        label="New Holland (PLM)",
        signature_columns=("plm_rate", "nh_field", "intelliview_id"),
        hint_columns=("grower", "farm", "field", "load", "operator", "task"),
        path_markers=("plm", "newholland", "new_holland", "intelliview",
                      "precision_land_management"),
        extensions=(".vy1", ".vy2", ".vy3"),
        default_units={"yield": "bu/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "isoxml", "csv"),
        notes="IntelliView IV and XCN are ISOBUS and consume TASKDATA.XML directly.",
    ),
    BrandProfile(
        key="bourgault",
        label="Bourgault",
        signature_columns=("bourgault_rate", "x35_rate", "tank_1_rate", "tank_2_rate"),
        hint_columns=("tank", "product", "section", "blockage", "shank"),
        path_markers=("bourgault", "x35", "x30", "apollo"),
        extensions=(),
        default_units={"rate": "lb/ac", "speed": "mph", "swath": "ft"},
        export_formats=("isoxml", "shapefile_rx", "csv"),
        notes="The X30/X35 monitors work with ISOXML and with a shapefile Rx.",
    ),
    BrandProfile(
        key="vaderstad",
        label="Väderstad",
        signature_columns=("vaderstad_rate", "e_control_rate", "econtrol_id"),
        hint_columns=("seed_rate", "fertilizer_rate", "section", "row"),
        path_markers=("vaderstad", "vaderstad", "e_control", "econtrol", "iseed",
                      "e_services", "vaderstad_ab"),
        extensions=(),
        default_units={"rate": "kg/ha", "speed": "km/h", "swath": "m"},
        export_formats=("isoxml", "shapefile_rx"),
        notes="E-Control and E-Services are ISOBUS — TASKDATA.XML is the official route.",
    ),
    BrandProfile(
        key="augmenta",
        label="Augmenta",
        signature_columns=(
            "augmenta_id", "session_id", "vigor", "vra_rate", "canopy",
            "crop_coverage", "biomass_index",
        ),
        hint_columns=("ndvi", "ndre", "index", "applied_rate", "machine_id", "nozzle"),
        path_markers=("augmenta", "augmenta_export", "field_analysis"),
        extensions=(".augmenta",),
        default_units={"rate": "l/ha", "speed": "km/h", "swath": "m"},
        export_formats=("geojson", "shapefile_rx", "csv"),
        notes=(
            "Augmenta exports GeoJSON/SHP per session, carrying vigour indices and "
            "the rate the vision system actually applied."
        ),
    ),
    BrandProfile(
        key="climate_fieldview",
        label="Climate FieldView",
        signature_columns=("fieldview_id", "cfv_rate"),
        hint_columns=("hybrid", "variety", "seed_rate", "yield", "elevation"),
        path_markers=("fieldview", "climate"),
        extensions=(),
        default_units={"yield": "bu/ac", "speed": "mph", "swath": "ft"},
        export_formats=("shapefile_rx", "csv"),
    ),
    BrandProfile(
        key="topcon",
        label="Topcon / Müller",
        signature_columns=("topcon_rate", "mueller_rate"),
        hint_columns=("task", "product", "section", "rate"),
        path_markers=("topcon", "mueller", "muller", "x35_topcon", "tap"),
        extensions=(),
        default_units={"rate": "kg/ha", "speed": "km/h", "swath": "m"},
        export_formats=("isoxml", "shapefile_rx"),
    ),
    BrandProfile(
        key="isoxml",
        label="Generic ISOBUS / ISOXML",
        signature_columns=("ddi", "tlg", "grd", "pdt", "ctp"),
        hint_columns=(),
        path_markers=("taskdata", "taskdata.xml", "grd", "tlg"),
        extensions=(".iso", ".xml"),
        default_units={"rate": "kg/ha", "speed": "km/h", "swath": "m"},
        export_formats=("isoxml",),
        notes="The ISO 11783-10 standard, accepted by most ISOBUS terminals.",
    ),
    BrandProfile(
        key="generic",
        label="Generic / not identified",
        export_formats=("shapefile_rx", "csv", "geojson", "isoxml"),
    ),
)

BRANDS_BY_KEY = {b.key: b for b in BRANDS}


def get_brand(key: str | None) -> BrandProfile:
    """Manufacturer profile by key, falling back to the generic one."""
    return BRANDS_BY_KEY.get(key or "", BRANDS_BY_KEY["generic"])


def detect_brand(
    columns: list[str] | None = None,
    path: str | None = None,
    extra_text: str | None = None,
) -> tuple[str, float]:
    """Identify the most likely manufacturer.

    Returns
    -------
    (key, confidence)
        Confidence runs from 0 to 1; below roughly 0.3 the result should be
        treated as a guess and confirmed by the user in the interface.
    """
    norm_cols = {normalize_name(c) for c in (columns or [])}
    haystack = normalize_name(path or "") + " " + normalize_name(extra_text or "")

    scores: dict[str, float] = {}
    for brand in BRANDS:
        if brand.key == "generic":
            continue
        score = 0.0
        for col in brand.signature_columns:
            if normalize_name(col) in norm_cols:
                score += 3.0
        for col in brand.hint_columns:
            if normalize_name(col) in norm_cols:
                score += 0.5
        for marker in brand.path_markers:
            if normalize_name(marker) in haystack:
                score += 2.5
        for ext in brand.extensions:
            if haystack.endswith(normalize_name(ext)) or normalize_name(ext).strip("_") in haystack:
                score += 2.0
        if score > 0:
            scores[brand.key] = score

    if not scores:
        return "generic", 0.0

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best] / total if total else 0.0
    # A low absolute score does not become certainty just by being the only one.
    confidence *= min(1.0, scores[best] / 5.0)
    return best, round(confidence, 3)


def brand_catalog() -> list[dict]:
    """Serializable catalogue for the interface."""
    return [
        {
            "key": b.key,
            "label": b.label,
            "export_formats": list(b.export_formats),
            "default_units": dict(b.default_units),
            "notes": b.notes,
        }
        for b in BRANDS
    ]
