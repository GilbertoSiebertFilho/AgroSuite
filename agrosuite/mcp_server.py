"""MCP server: driving AgroSuite by asking Claude.

AgroSuite already exposes everything it does over a local HTTP API, so this
server is a translation layer, not a second implementation. It speaks the
Model Context Protocol over stdin and stdout, and every tool call becomes one
HTTP request to the app running on this machine.

What that buys is the thing a graphical interface is bad at: saying what you
want in one sentence. "Load these three files, clean the yield map, join them
and tell me the optimum rate for canola at sixteen fifty a bushel" is a
sentence; as clicks it is five screens. The interface stays the place to look
at a map and judge a cleaning; this is the place to ask for work.

Two deliberate limits:

**Nothing here writes outside the app's own workspace.** Every tool maps to an
endpoint the interface itself uses, so the MCP surface can do no more than a
person clicking could.

**Writing to a USB drive asks first.** The tool reports what would be replaced
and refuses until the caller passes the names explicitly. An assistant that
silently overwrote last season's prescription would be worse than no
assistant.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# The one import from the app itself, and it is the app's own renderer
# rather than a second one: this server writes sentences about the same
# field the person has on screen, so it has to write them the same way.
from .core.units import Phrase

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "agrosuite"
SERVER_VERSION = "1.0.0"

#: AgroSuite moves to the next free port when 8765 is taken.
PORTS = range(8765, 8775)
TIMEOUT_S = 120

#: Slope at and above which the terrain summary calls ground steep, the
#: same threshold the analysis itself quotes: where erosion on bare soil
#: and machine stability both become a concern.
STEEP_PCT = 10.0


# ==========================================================================
# Talking to the app
# ==========================================================================

class App:
    """The AgroSuite server, found or started."""

    def __init__(self, autostart: bool = True) -> None:
        self.base: str | None = None
        self.autostart = autostart
        self._process: subprocess.Popen | None = None

    def _probe(self, base: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                return json.load(response).get("ok") is True
        except Exception:
            return False

    def find(self) -> str | None:
        if self.base and self._probe(self.base):
            return self.base
        for port in PORTS:
            candidate = f"http://127.0.0.1:{port}"
            if self._probe(candidate):
                self.base = candidate
                return candidate
        return None

    def ensure(self) -> str:
        base = self.find()
        if base:
            return base
        if not self.autostart:
            raise RuntimeError(
                "AgroSuite is not running. Start it with 'python -m agrosuite' "
                "and try again."
            )
        # Start it headless: the assistant does not need the browser window,
        # and opening one unasked would be rude.
        self._process = subprocess.Popen(
            [sys.executable, "-m", "agrosuite", "--no-browser"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            time.sleep(0.5)
            base = self.find()
            if base:
                return base
        raise RuntimeError(
            "AgroSuite was started but did not answer within 20 seconds. "
            "Check that its dependencies are installed."
        )

    def call(self, method: str, path: str, body: Any = None) -> Any:
        base = self.ensure()
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            base + path, data=data,
            headers={"Content-Type": "application/json"}, method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error).get("detail", error.reason)
            except Exception:
                detail = error.reason
            raise RuntimeError(str(detail)) from None


APP = App()


# ==========================================================================
# Tools
# ==========================================================================

def _text(value: Any) -> dict[str, Any]:
    body = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    return {"content": [{"type": "text", "text": body}]}


def tool_open_file(path: str, rate_unit: str = "", speed_unit: str = "",
                   length_unit: str = "", crop: str = "") -> Any:
    units = {}
    if rate_unit:
        units.update({"value": rate_unit, "target_rate": rate_unit,
                      "applied_rate": rate_unit})
    if speed_unit:
        units["speed_kmh"] = speed_unit
    if length_unit:
        # Every length in the file is in the one system the monitor was set
        # to, the GPS altitude included; declaring only the width would
        # leave the terrain analyser reading feet as metres.
        units.update({"swath_m": length_unit, "distance_m": length_unit,
                      "elev_m": length_unit})
    result = APP.call("POST", "/api/import/path", {
        "path": path, "source_units": units, "crop": crop or None,
    })
    preflight = result.get("preflight") or {}
    return {
        "dataset_id": result["id"],
        "label": result["label"],
        "rows": result["rows"],
        "monitor": result["meta"]["brand_label"],
        "operation": result["meta"]["operation_label"],
        "suggested_role": preflight.get("role_label"),
        "first_look": preflight.get("summary"),
        "findings": [
            f"[{f['level']}] {f['title']}: {f['detail']}"
            for f in preflight.get("findings", []) if f["level"] != "ok"
        ],
        "next_step": preflight.get("next_step", {}).get("label"),
    }


def tool_list_datasets() -> Any:
    listing = APP.call("GET", "/api/datasets")
    return [
        {
            "dataset_id": d["id"], "label": d["label"], "rows": d["rows"],
            "role": d.get("role"), "operation": d["meta"]["operation_label"],
            "monitor": d["meta"]["brand_label"],
        }
        for d in listing["datasets"]
    ]


def tool_describe_dataset(dataset_id: str) -> Any:
    detail = APP.call("GET", f"/api/datasets/{dataset_id}")
    return {
        "label": detail["label"], "rows": detail["rows"],
        "columns": detail["columns"], "area_ha": detail["area_ha"],
        "operation": detail["meta"]["operation_label"],
        "monitor": detail["meta"]["brand_label"],
        "notes": detail["meta"]["notes"],
        "statistics": detail["stats"],
        "first_look": (detail.get("reports_data") or {}).get("preflight", {}).get("summary"),
    }


def tool_project_status() -> Any:
    project = APP.call("GET", "/api/project")
    return {
        "stage": project["stage_label"],
        "goal": project["evaluation"]["label"],
        "ready": project["evaluation"]["ready"],
        "missing": [r["label"] for r in project["evaluation"]["missing"]],
        "layers": [
            {"dataset_id": l["dataset_id"], "label": l["label"], "role": l["role_label"]}
            for l in project["layers"]
        ],
        "next_action": project["next_action"],
    }


def tool_set_role(dataset_id: str, role: str) -> Any:
    project = APP.call("POST", "/api/project/role",
                       {"dataset_id": dataset_id, "role": role})
    return {"stage": project["stage_label"],
            "missing": [r["label"] for r in project["evaluation"]["missing"]]}


def tool_set_prices(crop_price_per_kg: float, input_cost_per_kg: float,
                    currency: str = "CAD", crop: str = "") -> Any:
    project = APP.call("POST", "/api/project/prices", {
        "crop_price": crop_price_per_kg, "input_cost": input_cost_per_kg,
        "currency": currency, "crop": crop or None,
    })
    return {"ready": project["evaluation"]["ready"],
            "missing": [r["label"] for r in project["evaluation"]["missing"]]}


def tool_clean(dataset_id: str, preset: str = "") -> Any:
    result = APP.call("POST", f"/api/datasets/{dataset_id}/clean",
                      {"preset": preset or None})
    report = result["report"]
    return {
        "clean_dataset_id": result["clean"]["id"],
        "removed_dataset_id": (result.get("removed") or {}).get("id"),
        "totals": report["totals"],
        "findings": [f"[{f['level']}] {f['text']}" for f in report["findings"]],
        "by_filter": [
            {"filter": s["label"], "removed": s["removed"]}
            for s in report["steps"] if s["removed"]
        ],
        "before_after": report["statistics"],
    }


def tool_join_layers(cell_m: float = 0, carry: list[str] | None = None) -> Any:
    result = APP.call("POST", "/api/project/join", {
        "cell_m": cell_m or None, "carry": carry or [],
    })
    return {
        "dataset_id": result["dataset"]["id"],
        "cells": result["report"]["cells"],
        "cell_m": result["cell_m"],
        "layers": result["report"]["layers"],
        "notes": result["report"]["notes"],
        "plan_vs_applied": result["report"].get("plan_vs_applied"),
    }


def tool_analyse_economics(dataset_id: str, crop_price_per_kg: float,
                           input_cost_per_kg: float, rate_column: str = "applied_rate",
                           value_column: str = "value", zone_column: str = "",
                           cell_m: float = 20.0, edge_margin_m: float = 6.0) -> Any:
    report = APP.call("POST", f"/api/datasets/{dataset_id}/difm", {
        "rate_column": rate_column, "value_column": value_column,
        "zone_column": zone_column or None,
        "crop_price": crop_price_per_kg, "input_cost": input_cost_per_kg,
        "cell_m": cell_m, "edge_margin_m": edge_margin_m,
    })
    out = {
        "model": report["chosen_model"]["label"],
        "r2": report["chosen_model"]["r2"],
        "cells": report["cells"],
        "rates_tested_kg_ha": report["rates_tested"],
        "economics_internal_units": report["economics"],
        "by_rate": report["by_rate"],
        "notes": report["notes"],
    }
    if report.get("zones"):
        out["by_zone"] = report["zones"]["by_zone"]
        out["uniform_vs_variable"] = report["zones"]["comparison"]
    return out


def tool_analyse_terrain(dataset_id: str, cell_m: float | None = None,
                         smooth_m: float | None = None,
                         contour_interval_m: float | None = None,
                         min_feature_height_m: float | None = None) -> Any:
    # Every option defaults to None rather than 0, because 0 is a real
    # request for two of them — 0 m of smoothing reads the raw surface —
    # and a 0 sentinel would quietly turn that back into "choose from the
    # data". Only what the caller set is sent; the app resolves the rest.
    body: dict[str, Any] = {"dataset_id": dataset_id}
    for name, value in (
        ("cell_m", cell_m), ("smooth_m", smooth_m),
        ("contour_interval_m", contour_interval_m),
        ("min_feature_height_m", min_feature_height_m),
    ):
        if value is not None:
            body[name] = value
    # A file with no usable elevation comes back as the analyser's own
    # sentence, which already names what to open instead; it travels to the
    # caller unchanged, like every other error here, rather than being
    # wrapped in a summary of nothing.
    summary = APP.call("POST", "/api/terrain/analyze", body)["summary"]
    # The set the app has just written this summary's findings in. Asked
    # for rather than assumed: it is the session's, and the session is the
    # one the person is looking at.
    units = APP.call("GET", "/api/units").get("display") or {}
    return _terrain_text(summary, units)


def _terrain_text(summary: dict[str, Any], units: dict[str, Any] | None = None) -> str:
    """The terrain summary as the paragraphs a person would read out.

    The JSON summary is a dozen nested tables, several of them histograms;
    handed over raw it would be relayed as a paraphrase, and a height whose
    unit was guessed is worse than no height. So every number is written
    here with its unit, and the findings, which are already the sentences a
    farmer acts on, travel verbatim.

    **Which unit is the app's decision, not this tool's.** The summary
    arrives with its findings already written in the set on screen, so a
    tool that wrote its own lines in metric would put two systems in one
    answer about one field — "60.5 ha of field ... over 34.6 ft of relief"
    — while the person asking has the second set in front of them. The
    alternative, metric throughout, would mean restating the findings
    against the screen: the assistant would then be the only voice in the
    room speaking metric, and the app's own promise is that what it says
    follows the picker. So everything here is written through the same
    :class:`~agrosuite.core.units.Phrase` the app writes with, from the
    same stored metric numbers, and the first line names the set once so
    a conversation can convert from it if it is asked to.

    A percentage and a bearing are the same in every unit set and are
    written plainly.
    """
    say = Phrase(units)
    grid = summary["grid"]
    elevation = summary["elevation"]
    slope = summary["slope"]
    trend = summary["trend"]
    features = summary["features"]
    wetness = summary["wetness"]

    lines = [
        f"Measurements below are in the units AgroSuite is showing: "
        f"{say.length_unit} for heights and distances, {say.area_unit} for area, "
        f"{say.volume_unit} for the water a hollow holds. The unit picker at the top "
        "of the app changes them, and this tool follows it.",
        f"{summary['character']['label'].capitalize()} field of "
        f"{say.area(grid['area_ha'])} with {say.length(elevation['relief_m'])} of total "
        f"relief (the fall from its highest ground, {say.length(elevation['max_m'])}, to "
        f"its lowest, {say.length(elevation['min_m'])}). {summary['character']['why']}",
    ]
    if elevation["level"]:
        lines.append(
            f"Nothing in this field rises or falls by more than the "
            f"{say.length(elevation['relief_floor_m'], 2)} of measurement noise, so it "
            "is reported as level: no contours, drainage lines, wet ground or features "
            "are drawn, because every one of them would be drawn from the noise."
        )

    if trend["drop_m"] >= 0.1:
        lines.append(
            f"Trend: the field falls {say.length(trend['drop_m'])} towards the "
            f"{trend['direction_label']} ({trend['direction_deg']:.0f} degrees "
            f"clockwise from north), an average gradient of "
            f"{trend['gradient_pct']:.1f} %."
        )
    else:
        lines.append("Trend: no consistent fall across the field.")

    lines.append(_terrain_features(features, say))

    steep_pct = sum(c["pct"] for c in slope["classes"] if (c["from_pct"] or 0.0) >= STEEP_PCT)
    steep_ha = sum(c["area_ha"] for c in slope["classes"] if (c["from_pct"] or 0.0) >= STEEP_PCT)
    lines.append(
        f"Steep ground: {steep_pct:.1f} % of the field ({say.area(steep_ha)}) is above "
        f"{STEEP_PCT:g} % slope; the slope averages {slope['mean_pct']:.1f} % and "
        f"reaches {slope['max_pct']:.1f} % at its steepest.\n"
        f"Likely wet: {say.area(wetness['wet_area_ha'])}, {wetness['wet_pct']:.0f} % of "
        "the field. (Every share is read inside the field's outermost ring of cells, "
        "whose slope leans on copied values; the ring is still drawn on the map "
        "layers.)"
    )

    lines.append("Findings:\n" + "\n".join(
        f"  [{f['level']}] {f['text']}" for f in summary["findings"]
    ))
    return "\n\n".join(lines)


def _terrain_features(features: dict[str, Any], say: Phrase) -> str:
    """Hills, low ground and closed depressions, each with where it is.

    The headings no longer name a unit — every quantity under them carries
    its own, in the set the reader is in — so there is nothing left to go
    stale when the picker moves.
    """
    blocks = []
    hills = features.get("hills") or []
    if hills:
        blocks.append("Hills (height above the ground around them):\n" + "\n".join(
            f"  {h['label']} in the {h['position']}: {say.length(h['height_m'])} over "
            f"{say.area(h['area_ha'])}, summit at {say.length(h['summit_m'])}."
            for h in hills
        ))
    lows = features.get("lows") or []
    if lows:
        blocks.append("Low ground (depth below the ground around it):\n" + "\n".join(
            f"  {l['label']} in the {l['position']}: {say.length(l['depth_m'])} below "
            f"over {say.area(l['area_ha'])}, bottom at {say.length(l['bottom_m'])}"
            + ("; part of it is closed, so water ponds there." if l["closed"]
               else "; it drains out, so water runs through rather than standing.")
            for l in lows
        ))
    depressions = features.get("depressions") or []
    if depressions:
        blocks.append(
            "Closed depressions (water ponds here until it spills):\n" + "\n".join(
                f"  {d['label']} in the {d['position']}: {say.area(d['area_ha'])}, up to "
                f"{say.length(d['max_depth_m'], 2)} deep, holding "
                f"{say.volume(d['volume_m3'])} before it spills at "
                f"{say.length(d['spill_m'])}."
                for d in depressions
            )
        )
    unlisted = int(features.get("depressions_unlisted") or 0)
    if unlisted:
        blocks.append(
            f"{unlisted} shallower "
            + ("hollow was" if unlisted == 1 else "hollows were")
            + " within the measurement noise (under "
            f"{say.length(features['depression_floor_m'], 2)} deep) and not listed; the "
            "ponding-depth layer still shows them."
        )
    if not blocks:
        return "No hill, low or closed depression stands out from the surrounding ground."
    return "\n".join(blocks)


def tool_terrain_zones(dataset_id: str, by: str = "landform",
                       kind: str = "points") -> Any:
    result = APP.call("POST", f"/api/terrain/{dataset_id}/zones",
                      {"by": by, "kind": kind})
    dataset = result["dataset"]
    return {
        "dataset_id": dataset["id"],
        "label": dataset["label"],
        "rows": dataset["rows"],
        "geometry": dataset["meta"]["geometry_type"],
        "zones": [
            f"{code}: {text}"
            for code, text in sorted(result["zone_labels"].items(), key=lambda kv: int(kv[0]))
        ],
        "next_step": (
            "The zones are a new dataset in the session like any other: give it a "
            "role with set_role if it is to go to the monitor, then export it and "
            "copy it with plan_usb_write and write_to_usb."
        ),
    }


def tool_design_trial(rates_kg_ha: list[float], boundary_dataset_id: str = "",
                      implement_width_m: float = 18.29, passes_per_strip: int = 2,
                      blocks: int = 4, buffer_m: float = 20.0) -> Any:
    result = APP.call("POST", "/api/design", {
        "rates": rates_kg_ha,
        "boundary_dataset_id": boundary_dataset_id or None,
        "implement_width_m": implement_width_m,
        "passes_per_strip": passes_per_strip,
        "blocks": blocks, "buffer_m": buffer_m,
    })
    return {"summary": result["summary"], "warnings": result["warnings"]}


def tool_list_usb_drives() -> Any:
    drives = APP.call("GET", "/api/usb")["drives"]
    return [
        {"path": d["path"], "label": d["label"], "removable": d["removable"],
         "free_mb": d["free_mb"],
         "already_on_it": [c["name"] for c in (d.get("contents") or [])]}
        for d in drives
    ]


def tool_plan_usb_write(folder: str, drive: str) -> Any:
    plan = APP.call("POST", "/api/usb/plan", {"folder": folder, "drive": drive})
    return {
        "to_copy": [item["name"] for item in plan["new"]],
        "would_replace": [item["name"] for item in plan["conflicts"]],
        "megabytes": plan["total_mb"],
        "fits": plan["fits"],
        "note": (
            "Nothing was written. To go ahead, call write_to_usb and pass the "
            "names to replace explicitly — anything not named is left alone."
        ),
    }


def tool_write_to_usb(folder: str, drive: str, replace: list[str] | None = None) -> Any:
    result = APP.call("POST", "/api/usb/write", {
        "folder": folder, "drive": drive, "replace": replace or [],
    })
    return {"copied": result["copied"],
            "left_alone": [s["name"] for s in result["skipped"]],
            "message": result["message"]}


def tool_validate_package(folder: str) -> Any:
    report = APP.call("POST", f"/api/validate?path={urllib.parse.quote(folder)}")
    return {
        "verdict": report["verdict"], "summary": report["summary"],
        "totals": report["totals"],
        "problems": [
            f"[{c['status']}] {c['item']}: {c['message']}"
            for group in report["groups"] for c in group["checks"]
            if c["status"] != "ok"
        ],
        "caveat": report["caveat"],
    }


#: The tool surface, with the schemas MCP needs to describe each one.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "open_file",
        "description": (
            "Open a monitor file in AgroSuite and report the preliminary "
            "analysis: what it is, what is missing and whether the units look "
            "right. Accepts shapefile, CSV, GeoJSON, ISOXML folders, John Deere "
            "cards and zips."
        ),
        "handler": tool_open_file,
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file or folder."},
                "rate_unit": {"type": "string",
                              "description": "Unit the yield or rate is written in, "
                                             "e.g. 'bu/ac' or 'lb/ac'. Leave empty to "
                                             "assume metric."},
                "speed_unit": {"type": "string", "description": "'mph' or 'km/h'."},
                "length_unit": {"type": "string",
                                "description": "'ft' or 'm': the unit of the swath "
                                               "width, distance and GPS altitude."},
                "crop": {"type": "string",
                         "description": "Crop key, needed for bushel units: canola, "
                                        "wheat, barley, oats, peas, corn, soybean."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_datasets",
        "description": "List everything currently loaded in AgroSuite.",
        "handler": tool_list_datasets,
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_dataset",
        "description": "Columns, statistics and provenance notes of one dataset.",
        "handler": tool_describe_dataset,
        "schema": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
        },
    },
    {
        "name": "project_status",
        "description": (
            "Where the project stands: which stage, which files hold which "
            "role, what is still missing and what to do next."
        ),
        "handler": tool_project_status,
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_role",
        "description": (
            "Say what a dataset is for: yield, as_applied, plan, vigor, "
            "boundary, guidance, soil or other."
        ),
        "handler": tool_set_role,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "role": {"type": "string",
                         "enum": ["yield", "as_applied", "plan", "vigor",
                                  "boundary", "guidance", "soil", "other"]},
            },
            "required": ["dataset_id", "role"],
        },
    },
    {
        "name": "set_prices",
        "description": (
            "Set the crop price and input cost, both per kilogram, which is the "
            "app's internal unit. To convert: dollars per bushel divided by the "
            "crop's bushel weight in kg (canola 22.68, wheat 27.22, barley 21.77, "
            "corn 25.40); dollars per pound divided by 0.4536."
        ),
        "handler": tool_set_prices,
        "schema": {
            "type": "object",
            "properties": {
                "crop_price_per_kg": {"type": "number"},
                "input_cost_per_kg": {"type": "number"},
                "currency": {"type": "string"},
                "crop": {"type": "string"},
            },
            "required": ["crop_price_per_kg", "input_cost_per_kg"],
        },
    },
    {
        "name": "clean_dataset",
        "description": (
            "Run the cleaning pipeline and return the report: how much was "
            "removed, by which filter, and how the statistics moved. Presets: "
            "harvest, application, planting, vigor, minimal."
        ),
        "handler": tool_clean,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "preset": {"type": "string",
                           "enum": ["harvest", "application", "planting", "vigor", "minimal"]},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "join_layers",
        "description": (
            "Put the project's yield, as-applied and plan layers onto a shared "
            "grid, ready for analysis. Leave cell_m at 0 to let the app choose "
            "from the sparsest layer's density."
        ),
        "handler": tool_join_layers,
        "schema": {
            "type": "object",
            "properties": {
                "cell_m": {"type": "number"},
                "carry": {"type": "array", "items": {"type": "string"},
                          "description": "Extra columns to bring from the yield layer, "
                                         "such as a management zone."},
            },
        },
    },
    {
        "name": "analyse_economics",
        "description": (
            "Fit a yield response curve to the rates that were actually applied "
            "and report the economic optimum rate — the point where the next "
            "unit of input stops paying for itself — with the yield and the "
            "margin expected there, the agronomic maximum, which model was "
            "chosen and how well it fits, and the mean yield of every rate that "
            "was applied. Given a zone column it fits one curve per zone and "
            "compares a variable-rate map against the best single rate. Prices "
            "are per kilogram; rates in the result are kg/ha."
        ),
        "handler": tool_analyse_economics,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "crop_price_per_kg": {"type": "number"},
                "input_cost_per_kg": {"type": "number"},
                "rate_column": {"type": "string"},
                "value_column": {"type": "string"},
                "zone_column": {"type": "string"},
                "cell_m": {"type": "number"},
                "edge_margin_m": {"type": "number"},
            },
            "required": ["dataset_id", "crop_price_per_kg", "input_cost_per_kg"],
        },
    },
    {
        "name": "analyse_terrain",
        "description": (
            "Read the relief of a loaded dataset from its GPS altitude (or from "
            "a DEM) and describe it in plain sentences: the character of the "
            "field and its total relief, which way it falls and how steeply, the "
            "hills, the low ground and the closed depressions with where they "
            "are, how big they are and how much water a depression holds, the "
            "share above 10 % slope, the likely wet area, and the findings. "
            "Every option is in metres; the answer quotes metres, hectares, "
            "percent and cubic metres."
        ),
        "handler": tool_analyse_terrain,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "cell_m": {"type": "number",
                           "description": "Grid cell size in metres. Left out, the app "
                                          "takes half the swath width."},
                "smooth_m": {"type": "number",
                             "description": "Smoothing scale in metres. Left out, just "
                                            "enough to take out the measured altitude "
                                            "noise; 0 reads the raw surface."},
                "contour_interval_m": {"type": "number",
                                       "description": "Contour step in metres. Left out, "
                                                      "chosen from the relief."},
                "min_feature_height_m": {"type": "number",
                                         "description": "How far a hill must stand above "
                                                        "its surroundings, in metres, to "
                                                        "be reported. Left out, twice the "
                                                        "measured GPS noise."},
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "terrain_zones",
        "description": (
            "Turn the relief of an analysed dataset into a zone layer — by "
            "landform, slope class, elevation band or wetness — as one point "
            "per grid cell or one polygon per zone. Returns a new dataset id "
            "the export tools take. Run analyse_terrain on the dataset first."
        ),
        "handler": tool_terrain_zones,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string",
                               "description": "The dataset that was analysed, not the "
                                              "zones of an earlier run."},
                "by": {"type": "string",
                       "enum": ["landform", "slope_class", "elevation_bands", "wetness"]},
                "kind": {"type": "string", "enum": ["points", "polygons"]},
            },
            "required": ["dataset_id", "by"],
        },
    },
    {
        "name": "design_trial",
        "description": (
            "Lay out a randomized block strip trial over a field boundary. "
            "Rates in kg/ha, widths in metres."
        ),
        "handler": tool_design_trial,
        "schema": {
            "type": "object",
            "properties": {
                "rates_kg_ha": {"type": "array", "items": {"type": "number"}},
                "boundary_dataset_id": {"type": "string"},
                "implement_width_m": {"type": "number"},
                "passes_per_strip": {"type": "integer"},
                "blocks": {"type": "integer"},
                "buffer_m": {"type": "number"},
            },
            "required": ["rates_kg_ha"],
        },
    },
    {
        "name": "list_usb_drives",
        "description": "Drives a package could be copied to, and what is already on them.",
        "handler": tool_list_usb_drives,
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "plan_usb_write",
        "description": (
            "Say what copying a package to a drive would replace. Writes "
            "nothing. Always call this before write_to_usb."
        ),
        "handler": tool_plan_usb_write,
        "schema": {
            "type": "object",
            "properties": {"folder": {"type": "string"}, "drive": {"type": "string"}},
            "required": ["folder", "drive"],
        },
    },
    {
        "name": "write_to_usb",
        "description": (
            "Copy a package to a drive. Anything already on the drive is left "
            "alone unless its name is passed in 'replace'."
        ),
        "handler": tool_write_to_usb,
        "schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string"}, "drive": {"type": "string"},
                "replace": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["folder", "drive"],
        },
    },
    {
        "name": "validate_package",
        "description": (
            "Check a package already on disk against the known causes of "
            "rejection on a monitor."
        ),
        "handler": tool_validate_package,
        "schema": {
            "type": "object",
            "properties": {"folder": {"type": "string"}},
            "required": ["folder"],
        },
    },
]

TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}


# ==========================================================================
# The protocol
# ==========================================================================

def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one JSON-RPC message, or ``None`` for a notification."""
    method = message.get("method")
    message_id = message.get("id")

    if method == "initialize":
        return _result(message_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "ping":
        return _result(message_id, {})

    if method == "tools/list":
        return _result(message_id, {
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["schema"],
                }
                for tool in TOOLS
            ]
        })

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            return _error(message_id, -32602, f"No such tool: {name}")
        try:
            value = tool["handler"](**(params.get("arguments") or {}))
        except TypeError as exc:
            return _result(message_id, {
                **_text(f"Wrong arguments for {name}: {exc}"), "isError": True,
            })
        except Exception as exc:
            return _result(message_id, {**_text(str(exc)), "isError": True})
        return _result(message_id, _text(value))

    if message_id is None:
        return None
    return _error(message_id, -32601, f"Unsupported method: {method}")


def _result(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def main() -> int:
    """Read messages from stdin, write answers to stdout, one JSON per line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            response = handle(message)
        except Exception as exc:  # the loop must survive any single message
            response = _error(message.get("id"), -32603, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
