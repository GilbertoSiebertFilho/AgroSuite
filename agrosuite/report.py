"""One-page printable report of what the app found in a dataset.

The screen is where the work happens; the decision is often taken somewhere
else — with an agronomist, at the kitchen table, in a meeting about next
season's inputs. A single page that can be printed or e-mailed carries the
result there without carrying the app along. One page is a constraint, not a
target: a report nobody reads past the first page might as well stop there.

Three rules keep the page honest:

* **Every number is converted the way the screen converts it.** The session
  stores metric; the report speaks whatever unit set the user picked, using
  the same factors as the interface, so the printed optimum is the optimum
  they saw. A report in kg/ha handed to someone who works in lb/ac would be
  misread, and a report that disagrees with the screen would be distrusted.
* **Only the sections that apply.** A dataset that has had a first look but
  no cleaning gets no cleaning section, not an empty one. Nothing here ever
  fails because a section is missing.
* **It says what it does not claim.** The curve describes this field, this
  season, these prices. The closing note mirrors the app's own tone: the
  known causes of error have been looked at; that is not a guarantee.

The app's own wording is set in the PDF standard font (Helvetica), which
every viewer carries and which covers the characters the app uses —
currency signs, degrees, the squared in R². The lines that carry the user's
words — the heading, the subtitle and the footer, where the project name and
the dataset label go — are set in a Unicode TrueType font found on the
machine when there is one: a standard font stops at WinAnsi, and a project
named in Cyrillic or Chinese would print as a row of squares. When no such
font loads, those lines fall back to Helvetica and the page is still built.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .core import units as units_mod
from .core.dataset import OPERATION_LABELS

#: Section keys, in the order they appear on the page. 'difm' is the key the
#: economic report is stored under — the wire name kept so that a project file
#: or a stored report written before the rename still opens; the heading on
#: the page reads "Economic report". 'terrain' sits where its tab does, next
#: to the first look: it describes the ground the other sections happened on.
SECTIONS = ("header", "preflight", "terrain", "clean", "difm", "caveat")

#: Origins the cleaning gives its two products, as the server registers them.
CLEANING_ORIGINS = ("clean", "clean_removed")

#: The interface's palette, so the printed page reads like the screen.
ACCENT = colors.HexColor("#2f7d4f")
ACCENT_SOFT = colors.HexColor("#e3f0e8")
WARN = colors.HexColor("#b8791c")
WARN_SOFT = colors.HexColor("#fdf2e0")
DANGER = colors.HexColor("#b33a3a")
DANGER_SOFT = colors.HexColor("#fbeaea")
TEXT = colors.HexColor("#1b241b")
MUTED = colors.HexColor("#5d6a5d")
FAINT = colors.HexColor("#8a958a")
BORDER = colors.HexColor("#d7ddd7")
BG = colors.HexColor("#f4f6f4")

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

#: Unicode TrueType fonts for the lines that carry the user's own words, in
#: order of preference: the name to register, the regular file, and the bold
#: files it may come with (Windows and macOS name Arial's bold differently).
#: DejaVu and Liberation cover Cyrillic, Greek and the accented Latin that
#: WinAnsi lacks; Arial is what Windows and macOS have; Vera ships inside
#: reportlab, so it is always there, but it covers Latin only.
UNICODE_FONTS = (
    ("DejaVuSans", "DejaVuSans.ttf", ("DejaVuSans-Bold.ttf",)),
    ("LiberationSans", "LiberationSans-Regular.ttf", ("LiberationSans-Bold.ttf",)),
    ("Arial", "arial.ttf", ("arialbd.ttf", "Arial Bold.ttf")),
    ("Vera", "Vera.ttf", ("VeraBd.ttf",)),
)

#: Wording the first look uses on screen, and what to print in its place.
#: A finding's action is written for the panel it appears in, where the
#: button that applies the proposed units sits right under it; on paper
#: there is no button, so the sentence has to say where it is instead.
ON_PAPER = (
    ("the button below", "the first look on the Data tab"),
)

#: What the report claims and what it leaves to the reader.
CAVEAT = (
    "What this page is: a summary of what AgroSuite computed from the files as they "
    "were loaded — the first look at the data, the cleaning it was given, and the "
    "response curve fitted to the strips. What it is not: a guarantee. The cleaning "
    "removed what the filters flagged, and whether that was headland and overlap or "
    "genuinely poor ground is something to confirm on the map. The optimum rate "
    "describes this field, in this season, at the prices entered; it is an "
    "extrapolation outside the rates actually tested, and it says nothing about a "
    "different year. Units follow the set chosen on screen when the report was made."
)


# ==========================================================================
# Units, the way the screen does them
# ==========================================================================

class _Units:
    """Conversions from the internal metric store to the user's unit set.

    Mirrors ``Units`` in the interface, factor for factor: the same
    ``unit_factor`` table, the same crop-dependent bushel, the same rule for
    which unit the main variable takes. Any divergence here would put two
    different numbers in front of the same person.
    """

    KEYS = ("yield_unit", "input_rate_unit", "area_unit", "length_unit",
            "speed_unit", "currency", "crop")
    GROUPS = {
        "yield_unit": "rate_mass",
        "input_rate_unit": "rate_mass",
        "area_unit": "area",
        "length_unit": "length",
        "speed_unit": "speed",
    }

    def __init__(self, units: dict[str, Any] | None, fallback_crop: str | None = None) -> None:
        preset = units_mod.UNIT_PRESETS[units_mod.DEFAULT_PRESET]
        given = dict(units or {})
        resolved: dict[str, Any] = {}
        for key in self.KEYS:
            value = given.get(key)
            if key == "crop" and not value:
                # The crop decides what a bushel weighs. The dataset's own crop
                # is a better guess than the preset's when the picker was empty.
                value = fallback_crop
            resolved[key] = str(value).strip() if value else preset[key]

        # Refusing an unknown unit beats printing numbers in a unit nobody
        # asked for. The message names what would have been accepted.
        for key, group in self.GROUPS.items():
            try:
                units_mod.unit_factor(group, resolved[key], resolved["crop"])
            except ValueError:
                valid = ", ".join(u["key"] for u in units_mod.UNIT_GROUPS[group]["units"])
                raise ValueError(
                    f"Unit '{resolved[key]}' is not a valid {key.replace('_', ' ')}. "
                    f"Use one of: {valid}."
                )
        self.prefs = resolved
        self.symbol = next(
            (c["symbol"] for c in units_mod.CURRENCIES if c["key"] == resolved["currency"]),
            resolved["currency"] or "$",
        )

    def _factor(self, group: str, key: str) -> float:
        return units_mod.unit_factor(group, self.prefs[key], self.prefs["crop"])

    # -- quantities ------------------------------------------------------
    def yld(self, v):
        return None if v is None else _f(v) / self._factor("rate_mass", "yield_unit")

    def rate(self, v):
        return None if v is None else _f(v) / self._factor("rate_mass", "input_rate_unit")

    def area(self, v):
        return None if v is None else _f(v) / self._factor("area", "area_unit")

    def length(self, v):
        return None if v is None else _f(v) / self._factor("length", "length_unit")

    def speed(self, v):
        return None if v is None else _f(v) / self._factor("speed", "speed_unit")

    def volume(self, v):
        """Cubic metres in the chosen length unit cubed.

        There is no volume group in the unit catalogue, because nothing else
        in the app measures one, so the length factor is cubed here: a cubic
        foot is 0.3048³ of a cubic metre. The unit is printed beside every
        number, since a pond's volume is the one figure nobody can check by
        eye.
        """
        if v is None:
            return None
        factor = self._factor("length", "length_unit")
        return _f(v) / (factor ** 3)

    def per_area(self, v):
        """Money per hectare (internal) to money per chosen area unit.

        A margin of 100 per hectare is 40.47 per acre: the factor is the one
        that takes the area unit to hectares, applied the other way round.
        """
        return None if v is None else _f(v) * self._factor("area", "area_unit")

    def mass_part_kg(self, rate_unit_key: str) -> float | None:
        """Kilograms in the numerator of a rate unit — 25.4 for a corn bu/ac."""
        numerator = str(rate_unit_key or "").split("/")[0]
        if numerator == "bu":
            return units_mod.bushel_kg(self.prefs["crop"])
        return units_mod.MASS_TO_KG.get(numerator)

    def price(self, per_kg, rate_unit_key: str):
        """A price per kg (internal) to a price per selling unit of the rate."""
        if per_kg is None:
            return None
        mass = self.mass_part_kg(rate_unit_key)
        return _f(per_kg) * mass if mass else _f(per_kg)

    def numerator(self, rate_unit_key: str) -> str:
        return str(rate_unit_key or "").split("/")[0]

    # -- labels ----------------------------------------------------------
    @property
    def yield_unit(self) -> str:
        return self.prefs["yield_unit"]

    @property
    def rate_unit(self) -> str:
        return self.prefs["input_rate_unit"]

    @property
    def area_unit(self) -> str:
        return self.prefs["area_unit"]

    @property
    def volume_unit(self) -> str:
        return f"{self.prefs['length_unit']}³"

    @property
    def length_unit(self) -> str:
        return self.prefs["length_unit"]

    def for_column(self, column: str, operation: str | None) -> tuple[Callable, str]:
        """Conversion and label for a canonical column, as ``Units.forColumn``.

        The main variable of a harvest is a yield; of anything else it is an
        input rate, and lb/ac of fertilizer printed as bu/ac of grain would
        be a gross error.
        """
        is_yield = operation is None or operation == "harvest"
        if column == "value":
            return (self.yld, self.yield_unit) if is_yield else (self.rate, self.rate_unit)
        if column in ("target_rate", "applied_rate"):
            return self.rate, self.rate_unit
        if column == "speed_kmh":
            return self.speed, self.prefs["speed_unit"]
        if column in ("swath_m", "distance_m", "elev_m"):
            return self.length, self.length_unit
        if column == "moisture_pct":
            return (lambda v: _f(v)), "%"
        return (lambda v: _f(v)), ""

    def money(self, v, decimals: int = 2) -> str:
        value = _f(v)
        if value is None:
            return "—"
        return f"{self.symbol} {_num(value, decimals)}"


# ==========================================================================
# Small formatting helpers
# ==========================================================================

def _f(value) -> float | None:
    """A finite float, or ``None`` for anything that cannot be printed."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _num(value, decimals: int | None = None) -> str:
    """Format as the interface does: decimals proportional to magnitude."""
    number = _f(value)
    if number is None:
        return "—"
    if decimals is None:
        magnitude = abs(number)
        decimals = 0 if magnitude == 0 else 0 if magnitude >= 1000 else 1 if magnitude >= 100 else 2 if magnitude >= 1 else 3
    return f"{number:,.{decimals}f}"


def _t(text: Any) -> str:
    """Text safe for Paragraph markup and for the standard fonts.

    The narrow no-break space the preliminary check uses as a thousands
    separator is not in the PDF standard encoding and would vanish, gluing
    "12 553" into "12553"; the ordinary no-break space renders and still
    keeps the number on one line.
    """
    return escape(str(text)).replace(" ", " ")


def _pct(value, decimals: int = 1) -> str:
    number = _f(value)
    return "—" if number is None else f"{_num(number, decimals)}%"


def _on_paper(text: Any) -> str:
    """Screen wording turned into paper wording, then made safe for markup."""
    text = str(text or "")
    for on_screen, on_paper in ON_PAPER:
        text = text.replace(on_screen, on_paper)
    return _t(text)


# ==========================================================================
# A font for the user's own words
# ==========================================================================

def _ttf_index() -> dict[str, Path]:
    """Every TrueType file under reportlab's font search path, by lower-cased name.

    reportlab searches the same directories, but only one level deep, and
    the distributions keep the fonts a level down — ``truetype/dejavu`` on
    Debian, ``TTF`` on Arch, ``Supplemental`` on macOS. One walk, kept as a
    map, answers for every candidate without walking again. Names are
    compared lower-cased because Windows keeps ``arial.ttf`` and macOS
    ``Arial.ttf``.
    """
    index: dict[str, Path] = {}
    for folder in rl_config.TTFSearchPath:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*.[tT][tT][fF]"):
                index.setdefault(path.name.lower(), path)
        except OSError:  # a folder that cannot be read is a folder without fonts
            continue
    return index


_FONTS_LOCK = threading.Lock()
_fonts: tuple[str, str] | None = None


def unicode_fonts() -> tuple[str, str]:
    """The (regular, bold) font names for the lines that carry the user's words.

    The first font in :data:`UNICODE_FONTS` that is on the disk and that
    reportlab can read is registered and returned; when none is, the
    standard Helvetica pair, so the page is still built. The search and the
    registration happen once per process, under a lock: two reports printed
    at the same moment must not race to register the same name. A bold file
    that is missing or unreadable costs only the bold — the regular face
    stands in for it rather than losing the font.
    """
    global _fonts
    with _FONTS_LOCK:
        if _fonts is None:
            _fonts = _register_unicode_fonts()
        return _fonts


def _register_unicode_fonts() -> tuple[str, str]:
    index = _ttf_index()
    for name, regular_file, bold_files in UNICODE_FONTS:
        regular = index.get(regular_file.lower())
        if regular is None:
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, str(regular)))
        except Exception:  # a font file reportlab cannot parse is as good as absent
            continue
        bold_name = name
        for bold_file in bold_files:
            bold = index.get(bold_file.lower())
            if bold is None:
                continue
            try:
                pdfmetrics.registerFont(TTFont(f"{name}-Bold", str(bold)))
            except Exception:
                continue
            bold_name = f"{name}-Bold"
            break
        # Paragraph markup (<b>) resolves through the family; without it a
        # tag inside a registered TrueType font is an error, not a style.
        pdfmetrics.registerFontFamily(name, normal=name, bold=bold_name,
                                      italic=name, boldItalic=bold_name)
        return name, bold_name
    return FONT, FONT_BOLD


# ==========================================================================
# Styles and building blocks
# ==========================================================================

def _styles(user_fonts: tuple[str, str] = (FONT, FONT_BOLD)) -> dict[str, ParagraphStyle]:
    """The page's styles; ``user_fonts`` is the pair for the user's own words."""
    user_font, user_font_bold = user_fonts
    base = ParagraphStyle("base", fontName=FONT, fontSize=7.6, leading=9.6, textColor=TEXT)
    return {
        "base": base,
        "title": ParagraphStyle("title", parent=base, fontName=user_font_bold, fontSize=13,
                                leading=16, spaceAfter=1),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontName=user_font, fontSize=8.4,
                                   leading=10.5, textColor=MUTED),
        "brand": ParagraphStyle("brand", parent=base, fontSize=7, leading=9,
                                textColor=FAINT, alignment=TA_RIGHT),
        "h2": ParagraphStyle("h2", parent=base, fontName=FONT_BOLD, fontSize=9.2,
                             leading=11, spaceBefore=5, textColor=TEXT),
        "muted": ParagraphStyle("muted", parent=base, fontSize=6.9, leading=8.6, textColor=MUTED),
        "note": ParagraphStyle("note", parent=base, fontSize=7.2, leading=9),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7, leading=8.4),
        "cell_r": ParagraphStyle("cell_r", parent=base, fontSize=7, leading=8.4,
                                 alignment=TA_RIGHT),
        "head": ParagraphStyle("head", parent=base, fontName=FONT_BOLD, fontSize=6.8,
                               leading=8.2, textColor=MUTED),
        "head_r": ParagraphStyle("head_r", parent=base, fontName=FONT_BOLD, fontSize=6.8,
                                 leading=8.2, textColor=MUTED, alignment=TA_RIGHT),
        "stat_k": ParagraphStyle("stat_k", parent=base, fontSize=6.4, leading=7.8,
                                 textColor=MUTED),
        "stat_v": ParagraphStyle("stat_v", parent=base, fontName=FONT_BOLD, fontSize=10.5,
                                 leading=12.5),
        "stat_d": ParagraphStyle("stat_d", parent=base, fontSize=6.4, leading=7.8,
                                 textColor=MUTED),
        "caveat": ParagraphStyle("caveat", parent=base, fontSize=6.6, leading=8.4,
                                 textColor=MUTED),
    }


def _heading(text: str, st: dict) -> list:
    return [
        Paragraph(_t(text), st["h2"]),
        HRFlowable(width="100%", thickness=0.6, color=ACCENT, spaceBefore=1, spaceAfter=3),
    ]


def _note(text: str, level: str, st: dict, width: float) -> Table:
    """A finding, coloured like the interface's ``.note`` boxes."""
    bar, soft = {
        "ok": (ACCENT, ACCENT_SOFT),
        "warning": (WARN, WARN_SOFT),
        "alert": (DANGER, DANGER_SOFT),
    }.get(level, (BORDER, BG))
    table = Table([[Paragraph(text, st["note"])]], colWidths=[width])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("LINEBEFORE", (0, 0), (0, -1), 2, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def _stats(items: list[tuple[str, str, str]], st: dict, width: float,
           per_row: int | None = None) -> Table:
    """Key figures in a grid, as the interface's ``.stat-grid``."""
    per_row = per_row or len(items)
    cells = [[
        Paragraph(_t(key), st["stat_k"]),
        Paragraph(_t(value), st["stat_v"]),
        Paragraph(_t(detail), st["stat_d"]),
    ] for key, value, detail in items]
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    rows[-1] += [""] * (per_row - len(rows[-1]))
    table = Table(rows, colWidths=[width / per_row] * per_row)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG),
        ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table


def _table(header: list[str], rows: list[list[str]], st: dict, widths: list[float],
           text_columns: int = 1) -> Table:
    """A data table in the interface's style: text left, numbers right."""
    def cell(value, column, is_head):
        style = ("head" if is_head else "cell") + ("" if column < text_columns else "_r")
        return Paragraph(_t(value), st[style])

    data = [[cell(v, i, True) for i, v in enumerate(header)]]
    data += [[cell(v, i, False) for i, v in enumerate(row)] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
    ]))
    return table


def _columns(left: list, right: list, widths: tuple[float, float]) -> Table:
    """Two blocks side by side, so the page holds a table next to a chart."""
    table = Table([[left, right]], colWidths=list(widths))
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, -1), 6),
        ("LEFTPADDING", (1, 0), (1, -1), 6),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


# ==========================================================================
# Sections
# ==========================================================================

def _header(entry, units: _Units, generated_at: str, project: dict, title: str | None,
            st: dict, width: float) -> list:
    meta = entry.dataset.meta
    project_name = str((project or {}).get("name") or "Untitled project")
    heading = title or project_name

    top = Table([[
        Paragraph(_t(heading), st["title"]),
        Paragraph(f"AgroSuite report<br/>{_t(generated_at)}", st["brand"]),
    ]], colWidths=[width * 0.72, width * 0.28])
    top.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    subtitle = _t(entry.label)
    if title and title != project_name:
        subtitle += f" · project: {_t(project_name)}"

    area_ha = 0.0
    try:
        area_ha = float(entry.dataset.area_ha())
    except Exception:  # a dataset without swath or distance has no worked area
        area_ha = 0.0
    crop = str(meta.crop).capitalize() if meta.crop else "—"
    operation = OPERATION_LABELS.get(meta.operation, meta.operation or "—")
    facts = [
        ("Field", meta.field_name or "—", ""),
        ("Crop", crop, ""),
        ("Monitor", meta.brand_label or "—", operation),
        ("Records", _num(len(entry.dataset), 0), ""),
        ("Area worked", _num(units.area(area_ha)) if area_ha > 0 else "—", units.area_unit),
    ]
    return [
        top,
        Paragraph(subtitle, st["subtitle"]),
        Spacer(1, 4),
        _stats(facts, st, width),
    ]


def _preflight_section(entry, units: _Units, st: dict, width: float,
                       next_step: bool = True) -> list:
    """The first look; ``next_step`` is False once the step it names is done.

    The first look runs when a dataset is registered, and its advice is
    written for a raw file: clean it. A clean copy, the removed records, or
    an original that has since been cleaned carry that same advice, and on
    paper it would sit right above the Cleaning section that answers it.
    What follows cleaning is the reader's decision, not the page's.

    It is **run again here**, over the dataset itself, in the unit set this
    page was asked for. The stored report's sentences were written for
    whatever was on screen when the file was opened, and a page that says
    "7.7 ha" beside a table in acres is the fault this whole path exists to
    remove. Re-reading the columns costs milliseconds and cannot drift from
    what it describes; if it fails for any reason, the stored sentences are
    printed rather than nothing.
    """
    from .core import preflight as _preflight

    report = entry.reports.get("preflight") or {}
    try:
        report = _preflight.run(entry.dataset, units.prefs)
    except Exception:
        pass
    verdict = str(report.get("verdict") or "warning")
    summary = str(report.get("summary") or "")
    findings = [f for f in (report.get("findings") or []) if f.get("level") != "ok"]

    flow = _heading("First look", st)
    flow.append(_note(f"<b>{_t(summary)}</b>", verdict, st, width))
    for finding in findings:
        text = f"<b>{_t(finding.get('title', ''))}.</b> {_t(finding.get('detail', ''))}"
        if finding.get("action"):
            text += f" <font color='#5d6a5d'>{_on_paper(finding['action'])}</font>"
        flow.append(Spacer(1, 2))
        flow.append(_note(text, str(finding.get("level") or ""), st, width))
    if not findings:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph("Every check came back clear.", st["muted"]))

    suggestion = (report.get("next_step") or {}) if next_step else {}
    if suggestion.get("label"):
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(
            f"Suggested next step: <b>{_t(suggestion['label'])}</b> — {_t(suggestion.get('why', ''))}",
            st["muted"],
        ))
    return flow


def _terrain_section(summary: dict, units: _Units, st: dict, width: float) -> list:
    """The relief: what kind of field it is, which way it falls, what stands
    out of it, and the sentences the analyser wrote about it.

    Compact by intention. The screen has room for twelve map layers, a
    histogram and a rose; the page has the four things that are acted on —
    the character, the fall, the features that hold water or stop machinery,
    and the share of the field in each slope class — plus the findings, which
    are what the reader takes to the kitchen table.

    The findings are **written again** here, in the unit set the page was
    asked for, from the numbers the stored summary carries — the same
    treatment the comparison below gets. Printing them as they came would
    put a sentence in metres next to a table in feet on a page that cannot
    be corrected afterwards.
    """
    from .terrain import analysis as _terrain

    summary = _terrain.restate(summary, units.prefs)
    elevation = summary.get("elevation") or {}
    trend = summary.get("trend") or {}
    slope = summary.get("slope") or {}
    wetness = summary.get("wetness") or {}
    features = summary.get("features") or {}
    character = summary.get("character") or {}

    flow = _heading("Relief", st)
    # A field with no fall worth the name is not given a direction: the
    # analyser reports a bearing for any residue of a plane fit, and "falls
    # to the north at 0.00 %" reads as information where there is none.
    gradient = _f(trend.get("gradient_pct")) or 0.0
    has_fall = gradient >= 0.05
    fall = str(trend.get("direction_label") or "—") if has_fall else "nowhere"
    fall_detail = (
        f"{_pct(gradient, 2)} · {_num(units.length(trend.get('drop_m')), 1)} "
        f"{units.length_unit} across" if has_fall else "no consistent fall"
    )
    flow.append(_stats([
        ("The field is", str(character.get("label") or "—").capitalize(), ""),
        ("Relief", _num(units.length(elevation.get("relief_m")), 1),
         f"{units.length_unit} · highest to lowest"),
        ("Mean slope", _pct(slope.get("mean_pct")),
         f"95% under {_pct(slope.get('p95_pct'))}"),
        ("Falls to the", fall, fall_detail),
        ("Likely wet", _num(units.area(wetness.get("wet_area_ha"))),
         f"{units.area_unit} · {_pct(wetness.get('wet_pct'))}"),
    ], st, width))

    # Hills, low ground and depressions in one table: the reader is looking
    # for what holds water and what stops the machine, not for three lists.
    rows = []
    for hill in features.get("hills") or []:
        rows.append([
            hill.get("label", ""), hill.get("position", ""),
            _num(units.area(hill.get("area_ha"))),
            f"+{_num(units.length(hill.get('height_m')), 1)}",
            f"top at {_num(units.length(hill.get('summit_m')), 1)} {units.length_unit}, "
            f"{_pct(hill.get('mean_slope_pct'))} slope",
        ])
    for low in features.get("lows") or []:
        rows.append([
            low.get("label", ""), low.get("position", ""),
            _num(units.area(low.get("area_ha"))),
            f"-{_num(units.length(low.get('depth_m')), 1)}",
            "part of it is closed: water ponds there" if low.get("closed")
            else "open: the water runs out",
        ])
    for hollow in features.get("depressions") or []:
        rows.append([
            hollow.get("label", ""), hollow.get("position", ""),
            _num(units.area(hollow.get("area_ha"))),
            f"-{_num(units.length(hollow.get('max_depth_m')), 2)}",
            f"holds {_num(units.volume(hollow.get('volume_m3')), 0)} {units.volume_unit} "
            f"to its spill at {_num(units.length(hollow.get('spill_m')), 1)} {units.length_unit}",
        ])

    left_width = width * 0.62
    right_width = width - left_width - 12
    if rows:
        left = [_table(
            ["Feature", "Where", f"Area ({units.area_unit})", f"Stands ({units.length_unit})",
             "And"], rows, st,
            [left_width * f for f in (0.16, 0.18, 0.13, 0.13, 0.4)], text_columns=2,
        )]
    else:
        left = [Paragraph(
            "No hill, hollow or closed depression stands out of the general fall of "
            "this field.", st["muted"])]
    unlisted = int(features.get("depressions_unlisted") or 0)
    if unlisted:
        left += [Spacer(1, 2), Paragraph(
            f"{unlisted} shallower hollow(s) were left out: they are under the "
            f"{_num(units.length(features.get('depression_floor_m')), 2)} "
            f"{units.length_unit} a feature has to stand out by, which is twice the "
            "noise in these readings.", st["muted"])]

    class_rows = [
        [cls.get("label", ""), _num(units.area(cls.get("area_ha"))), _pct(cls.get("pct"))]
        for cls in slope.get("classes") or []
    ]
    right = [_table(
        ["Slope class", f"Area ({units.area_unit})", "Share"],
        class_rows or [["—", "", ""]], st,
        [right_width * f for f in (0.45, 0.27, 0.23)],
    )]

    flow.append(Spacer(1, 4))
    flow.append(_columns(left, right, (left_width + 6, right_width + 6)))

    findings = summary.get("findings") or []
    warnings = [f for f in findings if f.get("level") == "warning"]
    rest = [f for f in findings if f.get("level") != "warning"]
    for finding in warnings:
        flow.append(Spacer(1, 3))
        flow.append(_note(_t(finding.get("text", "")), "warning", st, width))
    if rest:
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(
            "\u2022 " + "<br/>\u2022 ".join(_t(f.get("text", "")) for f in rest), st["note"]))
    return flow


def _terrain_yield_section(summary: dict, units: _Units, st: dict, width: float) -> list:
    """Yield against the relief: the findings, the elevation bands and the
    landform classes.

    The same intent as the relief section above it — compact, and only what
    is acted on. The screen has a chart with a quartile band under it and
    three tables; the page has the two tables that answer the question the
    relief raises (does the height cost me anything, and which part of the
    field is it), plus the findings, which carry the caveats. The slope and
    wetness splits stay on screen: on a page this wide a third table would
    cost the findings their room, and the landform classes already say
    where on the field the difference sits.

    The value follows the operation, as everywhere else: a harvest prints
    in the yield unit and anything else in the input-rate unit, because
    lb/ac of fertilizer printed as bu/ac of grain is a gross error. So do
    the findings: they are written again here, from the stored numbers, in
    the unit set the page was asked for, so the prose and the tables around
    it never quote the same figure in two different units.
    """
    values = summary.get("values") or {}
    overall = summary.get("overall") or {}
    points = summary.get("points") or {}
    convert, unit = units.for_column(
        str(values.get("column") or "value"), values.get("operation"))
    # The stored findings were written in whatever units the screen was in
    # when Compare was pressed; the page was asked for its own. They are
    # rewritten from the same numbers rather than converted, because a
    # sentence cannot be restated in another unit without re-deciding what
    # it says — the analyser is the one entitled to decide.
    from .terrain import yieldrelief as _yieldrelief

    written = _yieldrelief.findings(summary, units.prefs)

    flow = _heading("Yield against the relief", st)
    flow.append(_stats([
        ("Compared", _t(values.get("label") or "—"), "against the relief above"),
        ("Field average", _num(convert(overall.get("mean")), 1), unit),
        ("Spread", _pct(overall.get("cv_pct"), 0),
         f"of the average · {_num(overall.get('points'), 0)} readings"),
        ("Points matched", _num(points.get("matched"), 0),
         f"of {_num(points.get('total'), 0)} · {_num(points.get('off_grid'), 0)} off the field"),
    ], st, width))

    bands = summary.get("elevation_bands") or []
    band_rows = [
        [f"{_num(units.length(band.get('from_m')), 1)}–{_num(units.length(band.get('to_m')), 1)}",
         _num(units.area(band.get("area_ha")), 1),
         _num(convert(band.get("mean")), 1),
         _signed_pct(band.get("delta_pct")),
         _num(band.get("points"), 0)]
        for band in bands
    ]
    landforms = [cls for cls in summary.get("landforms") or [] if cls.get("points")]
    landform_rows = [
        [cls.get("label", ""),
         _num(convert(cls.get("mean")), 1),
         _signed_pct(cls.get("delta_pct")),
         _num(cls.get("points"), 0)]
        for cls in landforms
    ]

    left_width = width * 0.56
    right_width = width - left_width - 12
    left = [_table(
        [f"Height ({units.length_unit})", f"Ground ({units.area_unit})",
         f"Mean ({unit})", "vs field", "Readings"],
        band_rows or [["—", "", "", "", ""]], st,
        [left_width * f for f in (0.28, 0.18, 0.2, 0.17, 0.17)],
    )]
    right = [_table(
        ["Landform", f"Mean ({unit})", "vs field", "Readings"],
        landform_rows or [["—", "", "", ""]], st,
        [right_width * f for f in (0.4, 0.22, 0.2, 0.18)],
    )]
    flow.append(Spacer(1, 4))
    flow.append(_columns(left, right, (left_width + 6, right_width + 6)))

    # The bands are equal-count, which is why the height ranges are uneven
    # and the ground each holds is not: worth one line, because a reader
    # comparing the first band's hectares with the last one's otherwise
    # reads the difference as a mistake.
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(
        "The bands hold the same number of readings each, so their height ranges differ "
        "and the ground in each is what the field actually has at that height.",
        st["muted"]))

    for finding in written:
        if finding.get("level") != "warning":
            continue
        flow.append(Spacer(1, 3))
        flow.append(_note(_t(finding.get("text", "")), "warning", st, width))
    rest = [f for f in written if f.get("level") != "warning"]
    if rest:
        flow.append(Spacer(1, 3))
        flow.append(Paragraph(
            "\u2022 " + "<br/>\u2022 ".join(_t(f.get("text", "")) for f in rest), st["note"]))
    return flow


def _signed_pct(value, decimals: int = 1) -> str:
    """A difference from the field average, with its sign written in.

    A bare "3.2" in a column headed "vs field" is read as above average by
    everyone and as below average by no one; the plus sign is what makes
    the minus sign mean something.
    """
    number = _f(value)
    if number is None:
        return "—"
    return f"{'+' if number > 0 else ''}{_num(number, decimals)} %"


def _clean_section(report: dict, operation: str | None, units: _Units, st: dict,
                   width: float) -> list:
    """What the cleaning removed, and what each filter was set to.

    The step details are written again in the unit set the page was asked
    for, from the settings and measurements the stored report carries: "a
    20 ft strip in from the worked edge" on a page in feet, "6 m" on one in
    metres, and never one beside the other.
    """
    from .clean import pipeline as _clean_pipeline

    report = _clean_pipeline.restate(report, units.prefs, operation)
    totals = report.get("totals") or {}
    conv, unit = units.for_column(str(report.get("value_column") or "value"), operation)
    before = (report.get("statistics") or {}).get("before") or {}
    after = (report.get("statistics") or {}).get("after") or {}

    flow = _heading("Cleaning", st)
    for finding in report.get("findings") or []:
        flow.append(_note(_t(finding.get("text", "")), str(finding.get("level") or ""), st, width))
        flow.append(Spacer(1, 2))

    area_before, area_after = units.area(totals.get("area_ha_before")), units.area(totals.get("area_ha_after"))
    flow.append(_stats([
        ("Came in", _num(totals.get("input"), 0), "records"),
        ("Kept", _num(totals.get("kept"), 0), "records"),
        ("Removed", _num(totals.get("removed"), 0), f"records · {_pct(totals.get('removed_pct'))}"),
        ("Area worked", f"{_num(area_before)} → {_num(area_after)}", units.area_unit),
    ], st, width))

    for correction in report.get("corrections") or []:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(_t(correction), st["muted"]))

    # Before and after, in the unit the screen shows for this variable.
    def change(key):
        b, a = _f(before.get(key)), _f(after.get(key))
        if b is None or a is None or b == 0:
            return "—"
        delta = (a - b) / abs(b) * 100.0
        return f"{delta:+.1f}%"

    unit_suffix = f" ({unit})" if unit else ""
    compare_rows = [
        [f"Mean{unit_suffix}", _num(conv(before.get("mean"))), _num(conv(after.get("mean"))), change("mean")],
        [f"Median{unit_suffix}", _num(conv(before.get("median"))), _num(conv(after.get("median"))), change("median")],
        [f"SD{unit_suffix}", _num(conv(before.get("std"))), _num(conv(after.get("std"))), change("std")],
        ["Coeff. of variation (%)", _num(before.get("cv"), 1), _num(after.get("cv"), 1), change("cv")],
    ]
    left_width = width * 0.46
    compare = _table(
        ["Statistic", "Before", "After", "Change"], compare_rows, st,
        [left_width * 0.43, left_width * 0.2, left_width * 0.2, left_width * 0.17],
    )

    # Only the filters that took something out get a row. On paper the
    # question is "what was removed, and by what"; a filter that ran and found
    # nothing is still worth naming, but not worth a line of zeros each.
    total = _f(totals.get("input")) or 0.0
    steps = report.get("steps") or []
    ran = [s for s in steps if not s.get("skipped")]
    skipped = [s for s in steps if s.get("skipped")]
    active = [s for s in ran if (_f(s.get("removed")) or 0.0) > 0]
    idle = [s for s in ran if s not in active]
    filter_rows = [
        [s.get("label", s.get("key", "")), _num(s.get("removed"), 0),
         _pct((_f(s.get("removed")) or 0.0) / total * 100.0 if total else None),
         _num(s.get("remaining"), 0)]
        for s in active
    ] or [["No filter removed anything." if ran else "No filter was enabled.", "", "", ""]]
    right_width = width - left_width - 12
    filters = _table(
        ["Filter", "Removed", "% of total", "Left"], filter_rows, st,
        [right_width * 0.46, right_width * 0.18, right_width * 0.18, right_width * 0.18],
    )
    right: list = [filters]
    if idle:
        names = ", ".join(str(s.get("label", s.get("key", ""))) for s in idle)
        right += [Spacer(1, 2), Paragraph(f"Ran and removed nothing: {_t(names)}.", st["muted"])]
    if skipped:
        names = "; ".join(
            f"{s.get('label', s.get('key', ''))}: {s.get('detail', '')}".rstrip(": ")
            for s in skipped
        )
        right += [Spacer(1, 2), Paragraph(f"Could not run — {_t(names)}", st["muted"])]

    flow.append(Spacer(1, 4))
    flow.append(_columns([compare], right, (left_width + 6, right_width + 6)))
    return flow


def _nice_ticks(low: float, high: float, target: int = 4) -> tuple[list[float], int]:
    """Round tick positions spanning ``[low, high]``, and the decimals to print."""
    if high <= low:
        high = low + (abs(low) * 0.1 or 1.0)
    raw = (high - low) / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = magnitude
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw:
            break
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    count = int(round((end - start) / step))
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return [start + i * step for i in range(count + 1)], decimals


def _response_drawing(report: dict, units: _Units, width: float, height: float) -> Drawing | None:
    """Yield and margin against rate, with the optimum marked.

    Drawn by hand rather than through a chart class so the two curves can
    share the rate axis while keeping their own scales: yield and money have
    nothing in common but the rate that produced them.
    """
    curve = [p for p in (report.get("curve") or []) if _f(p.get("rate")) is not None]
    if len(curve) < 2:
        return None
    economics = report.get("economics") or {}
    has_profit = any(_f(p.get("profit")) is not None for p in curve)

    rates = [units.rate(p["rate"]) for p in curve]
    yields = [units.yld(p["yield"]) for p in curve]
    profits = [units.per_area(p["profit"]) for p in curve] if has_profit else []
    observed = [
        (units.rate(r["rate"]), units.yld(r["mean_yield"]))
        for r in (report.get("by_rate") or [])
        if _f(r.get("rate")) is not None and _f(r.get("mean_yield")) is not None
    ]

    pad_left, pad_right, pad_bottom, pad_top = 34, 36 if has_profit else 10, 20, 14
    x0, x1 = pad_left, width - pad_right
    y0, y1 = pad_bottom, height - pad_top

    x_ticks, x_dec = _nice_ticks(min(rates), max(rates))
    y_values = yields + [y for _, y in observed]
    y_ticks, y_dec = _nice_ticks(min(y_values), max(y_values))
    p_ticks, p_dec = _nice_ticks(min(profits), max(profits)) if has_profit else ([0, 1], 0)

    def sx(v):
        return x0 + (v - x_ticks[0]) / (x_ticks[-1] - x_ticks[0]) * (x1 - x0)

    def sy(v):
        return y0 + (v - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) * (y1 - y0)

    def sp(v):
        return y0 + (v - p_ticks[0]) / (p_ticks[-1] - p_ticks[0]) * (y1 - y0)

    d = Drawing(width, height)
    label = dict(fontName=FONT, fontSize=5.8, fillColor=MUTED)

    # Grid and axes.
    for tick in y_ticks:
        y = sy(tick)
        d.add(Line(x0, y, x1, y, strokeColor=BORDER, strokeWidth=0.3))
        d.add(String(x0 - 3, y - 2, _num(tick, y_dec), textAnchor="end", **label))
    for tick in x_ticks:
        x = sx(tick)
        d.add(Line(x, y0, x, y0 - 2, strokeColor=MUTED, strokeWidth=0.4))
        d.add(String(x, y0 - 9, _num(tick, x_dec), textAnchor="middle", **label))
    if has_profit:
        for tick in p_ticks:
            d.add(String(x1 + 3, sp(tick) - 2, _num(tick, p_dec), textAnchor="start", **label))
    d.add(Line(x0, y0, x1, y0, strokeColor=MUTED, strokeWidth=0.5))
    d.add(Line(x0, y0, x0, y1, strokeColor=MUTED, strokeWidth=0.5))
    if has_profit:
        d.add(Line(x1, y0, x1, y1, strokeColor=MUTED, strokeWidth=0.5))

    # Axis titles: horizontal, at the top of each axis — a rotated title
    # would need a transform for no gain at this size.
    d.add(String(x0, y1 + 4, f"Yield ({units.yield_unit})", textAnchor="start",
                 fontName=FONT, fontSize=6, fillColor=ACCENT))
    if has_profit:
        d.add(String(x1, y1 + 4, f"Margin ({units.symbol}/{units.area_unit})", textAnchor="end",
                     fontName=FONT, fontSize=6, fillColor=WARN))
    d.add(String((x0 + x1) / 2, 2, f"Rate ({units.rate_unit})", textAnchor="middle", **label))

    # Curves: yield on the left scale, margin on the right.
    if has_profit:
        points = []
        for r, p in zip(rates, profits):
            points += [sx(r), sp(p)]
        d.add(PolyLine(points, strokeColor=WARN, strokeWidth=1.1))
    points = []
    for r, y in zip(rates, yields):
        points += [sx(r), sy(y)]
    d.add(PolyLine(points, strokeColor=ACCENT, strokeWidth=1.3))
    for r, y in observed:
        d.add(Circle(sx(r), sy(y), 1.7, fillColor=ACCENT, strokeColor=colors.white, strokeWidth=0.4))

    # The optimum, marked on the rate axis. The label sits low in the plot:
    # a response curve climbs towards its optimum, so the top is where the
    # curves are and the bottom is where the space is.
    optimum = units.rate(economics.get("optimum_rate"))
    if optimum is not None and x_ticks[0] <= optimum <= x_ticks[-1]:
        x = sx(optimum)
        d.add(Line(x, y0, x, y1, strokeColor=DANGER, strokeWidth=0.8, strokeDashArray=[2, 2]))
        anchor = "end" if x > (x0 + x1) / 2 else "start"
        d.add(String(x - 3 if anchor == "end" else x + 3, y0 + 4,
                     f"optimum {_num(optimum)} {units.rate_unit}", textAnchor=anchor,
                     fontName=FONT_BOLD, fontSize=6, fillColor=DANGER))
    return d


def _difm_section(report: dict, units: _Units, st: dict, width: float) -> list:
    economics = report.get("economics") or {}
    model = report.get("chosen_model") or {}
    has_profit = bool(economics)

    flow = _heading("Economic report — yield response to rate", st)
    r2 = _f(model.get("r2"))
    flow.append(_stats([
        ("Economic optimum rate", _num(units.rate(economics.get("optimum_rate"))), units.rate_unit),
        ("Expected yield", _num(units.yld(economics.get("yield_at_optimum"))), units.yield_unit),
        ("Margin at the optimum", units.money(units.per_area(economics.get("profit_at_optimum"))),
         f"per {units.area_unit}"),
        ("Agronomic maximum", _num(units.rate(economics.get("agronomic_maximum"))), units.rate_unit),
        ("Model", str(model.get("label") or "—"),
         f"R² {_num(r2, 3)} · RMSE {_num(units.yld(model.get('rmse')))} {units.yield_unit} · {_num(model.get('n'), 0)} cells"),
    ], st, width))

    if economics.get("at_range_limit"):
        flow.append(Spacer(1, 2))
        flow.append(_note(
            "The optimum landed at the edge of the tested range — the trial never showed "
            "the point of diminishing returns. Read it as “at least this much”.",
            "alert", st, width,
        ))
    if model.get("message"):
        flow.append(Spacer(1, 2))
        flow.append(_note(_t(model["message"]), "warning", st, width))
    notes = [str(n) for n in (report.get("notes") or []) if n]
    if notes:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(_t(" ".join(notes)), st["muted"]))

    details = []
    parameters = report.get("parameters") or {}
    if parameters:
        details.append(
            f"{_num(units.length(parameters.get('cell_m')), 0)} {units.length_unit} cells · "
            f"{_num(units.length(parameters.get('edge_margin_m')), 0)} {units.length_unit} edge margin"
        )
    prices = report.get("prices") or {}
    if has_profit and prices:
        details.append(
            f"prices used: crop {units.money(units.price(prices.get('crop_price'), units.yield_unit))} "
            f"per {units.numerator(units.yield_unit)}, input "
            f"{units.money(units.price(prices.get('input_cost'), units.rate_unit))} "
            f"per {units.numerator(units.rate_unit)}"
        )
    if details:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph(_t(" · ".join(details)).replace("\n", ""), st["muted"]))

    # Curve beside the by-rate table: the same information twice, in the two
    # forms people actually read — the shape, and the numbers.
    left_width = width * 0.5
    right_width = width - left_width - 12
    drawing = _response_drawing(report, units, left_width, 132)
    header = [f"Rate ({units.rate_unit})", "Cells", f"Yield ({units.yield_unit})"]
    widths = [right_width * 0.3, right_width * 0.2, right_width * 0.5]
    if has_profit:
        header.append(f"Margin/{units.area_unit}")
        widths = [right_width * 0.26, right_width * 0.16, right_width * 0.28, right_width * 0.3]
    rate_rows = []
    for row in report.get("by_rate") or []:
        cells = [_num(units.rate(row.get("rate"))), _num(row.get("n"), 0),
                 _num(units.yld(row.get("mean_yield")))]
        if has_profit:
            cells.append(units.money(units.per_area(row.get("mean_profit"))))
        rate_rows.append(cells)
    right: list = [_table(header, rate_rows or [["—"] * len(header)], st, widths, text_columns=0)]

    # Uniform against variable rate goes under the by-rate table, where the
    # chart leaves room, so the zone table below stays next to its verdict.
    zones = report.get("zones") or {}
    by_zone = zones.get("by_zone") or []
    comparison = zones.get("comparison") or {}
    if comparison.get("reading"):
        right += [Spacer(1, 4), _stats([
            ("Best single rate", _num(units.rate(comparison.get("best_uniform_rate"))), units.rate_unit),
            ("Variable rate gain", units.money(units.per_area(comparison.get("gain_per_ha"))),
             f"per {units.area_unit}"),
            ("Margin at that rate", units.money(units.per_area(comparison.get("uniform_profit"))),
             f"per {units.area_unit}"),
            ("Margin with zones", units.money(units.per_area(comparison.get("variable_profit"))),
             f"per {units.area_unit}"),
        ], st, right_width, per_row=2)]

    flow.append(Spacer(1, 4))
    if drawing is not None:
        flow.append(_columns([drawing], right, (left_width + 6, right_width + 6)))
    else:
        flow += right

    if by_zone:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(f"<b>By zone · {_t(zones.get('zone_column', ''))}</b>", st["base"]))
        flow.append(Spacer(1, 2))
        header = ["Zone", "Cells", "Model", "R²", f"Optimum ({units.rate_unit})",
                  f"Yield ({units.yield_unit})", f"Margin/{units.area_unit}"]
        rows = []
        for zone in by_zone:
            if zone.get("error"):
                rows.append([zone.get("zone", ""), _num(zone.get("cells"), 0),
                             zone["error"], "", "", "", ""])
                continue
            rows.append([
                zone.get("zone", ""), _num(zone.get("cells"), 0), zone.get("model", "—"),
                _num(zone.get("r2"), 3), _num(units.rate(zone.get("optimum_rate"))),
                _num(units.yld(zone.get("yield_at_optimum"))),
                units.money(units.per_area(zone.get("profit_at_optimum"))),
            ])
        fractions = (0.14, 0.09, 0.19, 0.1, 0.16, 0.16, 0.16)
        flow.append(_table(header, rows, st, [width * f for f in fractions], text_columns=3))
        if comparison.get("reading"):
            gain = _f(comparison.get("gain_per_ha")) or 0.0
            flow.append(Spacer(1, 3))
            flow.append(_note(_t(comparison["reading"]), "ok" if gain > 0 else "warning", st, width))
    return flow


# ==========================================================================
# Entry point
# ==========================================================================

def build_pdf(
    entry,
    out_path: str | Path,
    units: dict[str, Any] | None,
    generated_at: str,
    project: dict[str, Any] | None,
    title: str | None = None,
) -> dict[str, Any]:
    """Write the one-page report for ``entry`` and say what went into it.

    Parameters
    ----------
    entry:
        A session entry: ``dataset``, ``label`` and ``reports`` are read. Only
        the reports present (``preflight``, ``terrain``, ``clean``, ``difm`` —
        the economic report's stored key) get a section.
    units:
        The unit set chosen on screen — ``yield_unit``, ``input_rate_unit``,
        ``area_unit``, ``length_unit``, ``speed_unit``, ``currency``, ``crop``.
        A missing key falls back to the app's default preset; an unknown unit
        is refused with a message naming the valid ones.
    generated_at:
        The timestamp printed on the page. It is passed in rather than read
        from the clock so the caller decides the timezone and the wording,
        and so a test can pin it.
    project:
        The session's project dictionary; only ``name`` is printed.
    title:
        Optional heading in place of the project name.

    Returns
    -------
    dict
        ``path``, ``sections`` (those actually printed, in order), ``pages``,
        ``size_bytes`` and the ``units`` the numbers were converted to.
    """
    out_path = Path(out_path)
    resolved = _Units(units, fallback_crop=entry.dataset.meta.crop)
    reports = entry.reports or {}
    user_font, _ = user_fonts = unicode_fonts()
    st = _styles(user_fonts)

    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=13 * mm, rightMargin=13 * mm, topMargin=11 * mm, bottomMargin=12 * mm,
        title=f"{title or (project or {}).get('name') or 'AgroSuite report'} — {entry.label}",
        author="AgroSuite", subject="Dataset report",
    )
    width = doc.width
    story: list = []
    sections: list[str] = []

    story += _header(entry, resolved, generated_at, project or {}, title, st, width)
    sections.append("header")

    if reports.get("preflight"):
        story.append(Spacer(1, 5))
        already_cleaned = entry.origin in CLEANING_ORIGINS or bool(reports.get("clean"))
        story.append(KeepTogether(_preflight_section(
            entry, resolved, st, width, next_step=not already_cleaned,
        )))
        sections.append("preflight")

    if reports.get("terrain"):
        story.append(Spacer(1, 5))
        story += _terrain_section(reports["terrain"], resolved, st, width)
        sections.append("terrain")

    # The comparison belongs under the relief that raised the question, and
    # only when both have been run: without the relief above it, a table of
    # yield by elevation band has nothing to be read against.
    if reports.get("terrain") and reports.get("terrain_yield"):
        story.append(Spacer(1, 5))
        story += _terrain_yield_section(reports["terrain_yield"], resolved, st, width)
        sections.append("terrain_yield")

    if reports.get("clean"):
        story.append(Spacer(1, 5))
        story += _clean_section(reports["clean"], entry.dataset.meta.operation, resolved, st, width)
        sections.append("clean")

    if reports.get("difm"):
        story.append(Spacer(1, 5))
        story += _difm_section(reports["difm"], resolved, st, width)
        sections.append("difm")

    if len(sections) == 1:
        story.append(Spacer(1, 6))
        story.append(_note(
            "No analysis has been run on this dataset yet. Read its relief, clean it, "
            "or run the economic analysis, and the report will carry the results.",
            "", st, width,
        ))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.4, color=BORDER, spaceAfter=3))
    story.append(Paragraph(_t(CAVEAT), st["caveat"]))
    sections.append("caveat")

    # Canvas strings take no markup, so the footer is not escaped — only the
    # separator the standard fonts lack is swapped, as in every paragraph.
    footer_text = f"AgroSuite · {entry.label} · {generated_at}".replace(" ", " ")

    def on_page(canvas, document):
        canvas.saveState()
        canvas.setFont(user_font, 6.2)
        canvas.setFillColor(FAINT)
        canvas.drawString(document.leftMargin, 7 * mm, footer_text)
        canvas.drawRightString(document.pagesize[0] - document.rightMargin, 7 * mm,
                               f"page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return {
        "path": str(out_path),
        "sections": sections,
        "pages": int(doc.page),
        "size_bytes": out_path.stat().st_size,
        "units": dict(resolved.prefs),
    }
