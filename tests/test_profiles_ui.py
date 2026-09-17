"""The machine-profile pickers in the interface, checked against the contract.

The interface is vanilla JavaScript and has no unit tests of its own; what
can be checked here without a browser is the contract it relies on. Every
field the page POSTs must be one the route accepts, every form field it
fills must be one the tabs draw, and the dialog it opens must exist in the
page. A rename on either side would otherwise only show up in a browser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrosuite.app.routes import profiles as profiles_routes
from agrosuite.clean import steps as clean_steps

STATIC = Path(__file__).resolve().parents[1] / "agrosuite" / "app" / "static"
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")


def _block(name: str) -> str:
    """The source of one App method, up to its closing brace at column two."""
    match = re.search(rf"^  (?:async )?{name}\(", APP_JS, flags=re.MULTILINE)
    assert match, f"App.{name} is not defined"
    end = APP_JS.index("\n  },\n", match.start())
    return APP_JS[match.start():end]


def test_dialog_and_its_buttons_are_in_the_page():
    for node_id in ("dlg-machine", "machine-body", "btn-machine-suggest", "btn-machine-save"):
        assert f'id="{node_id}"' in INDEX, f"index.html has no #{node_id}"


def test_every_tab_draws_and_binds_its_picker():
    for scope in ("clean", "design", "package"):
        assert f'this.machinePicker("{scope}")' in APP_JS, f"no picker for {scope}"
        assert f'this.bindMachinePicker("{scope}")' in APP_JS, f"picker for {scope} never bound"


def test_the_page_posts_only_fields_the_route_accepts():
    """A key the route ignores would be a silently dropped setting."""
    body = _block("saveMachine")
    body = body[body.index("const body = {"):body.index("};", body.index("const body = {"))]
    # `name,` is the shorthand for `name: name`.
    posted = set(re.findall(r"^\s+(\w+)(?::|,\s*$)", body, flags=re.MULTILINE))
    accepted = set(profiles_routes.ProfileRequest.model_fields)
    assert posted == accepted, f"posted {sorted(posted)} vs accepted {sorted(accepted)}"


def test_the_page_calls_the_three_endpoints():
    routes = {route.path for route in profiles_routes.router.routes}
    assert "/api/profiles" in routes and "/api/profiles/suggest" in routes
    assert 'this.api("/api/profiles")' in APP_JS
    assert 'this.api("/api/profiles/suggest"' in APP_JS
    assert 'this.api("/api/profiles", { method: "POST"' in APP_JS
    # Names carry spaces; an unencoded one would split the path.
    assert "`/api/profiles/${encodeURIComponent(name)}`" in APP_JS


def test_pickers_fill_the_fields_the_tabs_draw():
    """The ids a profile is poured into are the ids the forms are read from."""
    scope = _block("machineScope")
    filled = set(re.findall(r'put\("([\w-]+)"', scope))
    assert filled == {"clean-delay", "p-speed_range-min", "p-speed_range-max",
                      "design-width", "design-passes"}
    elsewhere = APP_JS.replace(scope, "")
    for node_id in ("clean-delay", "design-width", "design-passes", "pkg-monitor"):
        assert f'"{node_id}"' in elsewhere, f"no form field draws #{node_id}"
    # The speed inputs are drawn per filter step as `p-${step.key}-${key}`, and
    # the checkbox as `en-${step.key}`; the step itself must carry min and max.
    assert "`p-${step.key}-${key}`" in elsewhere and "`en-${step.key}`" in elsewhere
    speed = next(cls for cls in clean_steps.STEP_CLASSES if cls.key == "speed_range")
    assert {"min", "max"} <= set(speed.defaults)


def test_units_cross_the_wire_in_metric():
    """Feet and mph are shown, never sent: every physical field is converted."""
    scope = _block("machineScope") + _block("saveMachine") + _block("renderMachineDialog")
    # Both readers go through the one conversion, so they cannot drift apart.
    assert "this.machineFieldMetric(" in _block("machineScope")
    assert "this.machineFieldMetric(" in _block("saveMachine")
    assert "this.machineMetric(" in _block("machineFieldMetric")
    assert "Units.toInternal[kind]" in _block("machineMetric")
    assert "Units.convert[kind]" in scope
    assert "toInternal.length" not in APP_JS[APP_JS.index("Machine profiles"):].replace(
        "Units.toInternal[kind]", "")  # no stray unconverted path
