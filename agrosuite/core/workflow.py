"""The guided workflow.

Work on field data happens in two tracks, and the app has to serve both.

**Plan a trial**, before the season: a boundary comes in, the rates, the plot
size, the replications and the direction are set, the layout and the AB lines
are generated, and the package goes to the monitor. Nothing here needs a
yield map, prices or cleaning.

**Evaluate a trial**, after harvest: the as-applied and the yield map come
in, with the plan if there is one; the units and the roles are reviewed; the
data is cleaned; the economics are worked out; the result goes back to the
machine.

This module holds both tracks as data, so the app can always answer three
questions without being asked: where am I, what would help next, and what
can I take away if I stop here.

Nothing in here blocks
----------------------
The stages are a *suggested* order, not a gate. Each one produces something
of its own — a first look, a clean copy with its report, a layout, an
economic report — and each of those is worth having on its own. So every
function that reports a stage marks it **done**, **current**, **skippable**
or **ahead**, and never "blocked": looking at the data and stopping is a
complete use of the app, so is cleaning a file and exporting the clean copy,
and so is opening an already clean file and going straight to the economics.

The rule that lets cleaning be skipped is read from the data rather than
from a tick box: the evaluate track moves on to ``analyse`` as soon as the
files have been reviewed — roles assigned, units declared — **and** either a
cleaning has been run or the goal's requirements are already satisfied (a
yield, a rate and prices are all present, so the analysis can run on the
data as it stands). A file that arrives already clean therefore reaches
``analyse`` without a cleaning report ever existing.

A *project* is the set of files that describe one field-and-season. The
economic analysis needs the rate that went out, the yield that came of it
and the prices that turn yield into money — usually three files (the plan,
the as-applied log and the harvest map), sometimes one that carries the rate
beside the yield. The workflow tracks which of those are present and names
the ones that are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The stages of the evaluate track, in the order they are suggested in.
#: Suggested, not required — see the module docstring.
STAGES = ("load", "review", "clean", "analyse", "export")

#: The stages of the plan track: a field comes in, the strips go out.
DESIGN_STAGES = ("load", "design", "export")

STAGE_LABELS = {
    "load": "Load the files",
    "review": "Review what came in",
    "clean": "Clean the data",
    "design": "Design the trial",
    "analyse": "Economics",
    "export": "Send it to the monitor",
}

STAGE_DESCRIPTIONS = {
    "load": "Open what came off the monitor. The app identifies the platform, "
            "normalizes the columns and runs a first look at the file.",
    "review": "Confirm the units and the role of each file before anything "
              "downstream inherits a wrong assumption.",
    "clean": "Remove overlap, headland turns and sensor faults, and read the "
             "report to judge whether the cleaning was sound. Worth doing on "
             "raw monitor data; a file that is already clean can skip it.",
    "design": "Lay the strips out over the field, draw the AB line along them "
              "and build the package the terminal will read.",
    "analyse": "Fit the yield response to the rates the trial applied and find "
               "the rate where the next unit of input stops paying for itself.",
    "export": "Build the package for the target monitor and check it before "
              "the stick leaves the computer.",
}

#: What each stage produces on its own, and how to take it away. A stage is
#: a place the work can legitimately end, so every one of them has an answer
#: to "and if I stop here?".
STAGE_OUTPUTS = {
    "load": {
        "produces": "the file read into the app, with its first look: coverage, "
                    "timeline, columns, data quality and whether the units are "
                    "what they seem",
        "export": "Print the report (PDF) on the Data tab, or export the file "
                  "as it stands from the Export tab.",
    },
    "review": {
        "produces": "each file named for what it is, in the units it is really in",
        "export": "Print the report (PDF) on the Data tab, or export the file "
                  "as it stands from the Export tab.",
    },
    "clean": {
        "produces": "a clean copy, the removed records with the reason for each, "
                    "and the report that says what came out and from where",
        "export": "Export the clean copy as a file, or print the cleaning "
                  "report, from the Cleaning tab.",
    },
    "design": {
        "produces": "the strip layout and the AB line that keeps the passes "
                    "aligned with it",
        "export": "Build the package for the monitor from the Trial design tab.",
    },
    "analyse": {
        "produces": "the response curve, the economic optimum rate and the "
                    "margin it earns",
        "export": "Print the economic report, or build a prescription from the "
                  "optimum, on the Economics tab.",
    },
    "export": {
        "produces": "the package laid out the way the target platform expects, "
                    "checked file by file",
        "export": "Copy it to the USB stick, or download it, from the Export tab.",
    },
}

#: The two tracks, and the goal each one sets. The track is derived from the
#: goal rather than stored beside it, so a project file written before tracks
#: existed still lands in the right one.
TRACKS = {
    "plan": {
        "label": "Plan a trial",
        "description": "Before the season: a boundary, the rates and the plot "
                       "size in, the layout and the AB lines out.",
        "goal": "trial_design",
        "stages": DESIGN_STAGES,
    },
    "evaluate": {
        "label": "Evaluate a trial",
        "description": "After harvest: the as-applied and the yield map in, the "
                       "optimum rate and what it is worth out.",
        "goal": "difm",
        "stages": STAGES,
    },
}


@dataclass
class Requirement:
    """One thing an analysis needs before it can run."""

    key: str
    label: str
    detail: str
    satisfied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "detail": self.detail,
            "satisfied": self.satisfied,
        }


#: What each kind of analysis needs. Roles come from
#: :mod:`agrosuite.core.preflight`.
#:
#: The keys are the wire names: they travel in project files, in stored
#: reports and over HTTP. ``difm`` kept its key when its label became
#: "Economic analysis", so a project saved before the rename still opens.
GOALS = {
    "trial_design": {
        "label": "Plan a trial",
        "description": "Randomized strips over the field, with the AB line that "
                       "keeps the passes on them.",
        "requires": [
            ("field", "A field to lay the strips on",
             "A boundary file is the truest outline; any layer with geometry "
             "works, and a yield map falls back to the hull of its points. You "
             "can also draw the field on the map."),
        ],
        "optional": [
            ("guidance", "A guidance line",
             "An AB line already flown in the field; the strips can be aligned "
             "to it instead of to the longest side."),
            ("terrain", "A terrain analysis",
             "Slope and landforms show where a strip would run across the "
             "slope, which mixes two different fields into one treatment."),
        ],
    },
    "difm": {
        # The key stays 'difm' — see the note above GOALS.
        "label": "Economic analysis",
        "description": "Yield response to rate, and the economic optimum.",
        "requires": [
            ("yield", "Yield map",
             "The harvest file. Without it there is no response to measure."),
            ("rate", "Applied rate",
             "The as-applied log, or the plan if that is all there is. The "
             "as-applied is better: the response follows what actually went out."),
            ("prices", "Crop price and input cost",
             "The optimum is an economic question. Without prices the app can "
             "only report the agronomic maximum, which is a different number."),
        ],
        "optional": [
            ("plan", "Plan / prescription",
             "Lets the app show how far the machine drifted from the map."),
            ("zone", "Management zones",
             "Turns one curve into one per zone, which is what justifies "
             "variable rate."),
        ],
    },
    "augmenta": {
        "label": "Augmenta analysis",
        "description": "Whether the vision system actually modulated the rate.",
        "requires": [
            ("vigor", "Augmenta session",
             "A file carrying the vigour index and the applied rate together."),
        ],
        "optional": [],
    },
    "export": {
        "label": "Send to a monitor",
        "description": "A package laid out the way the target platform expects.",
        "requires": [
            ("geometry", "Something with geometry",
             "A prescription, a trial layout or a field boundary."),
        ],
        "optional": [
            ("guidance", "AB lines",
             "Carried in the same ISOXML as the boundary and the prescription."),
        ],
    },
}

#: Optional requirements that are noise unless the project already holds the
#: thing they talk about. A terrain analysis helps place the strips, but
#: naming it to someone who has loaded no elevation layer is one more line of
#: "you could also…" in a panel that exists to say what is missing.
OPTIONAL_WHEN_ROLE_PRESENT = {"terrain"}


def track_of(goal: str) -> str:
    """Which track a goal belongs to.

    Everything that is not the trial design is work on data that already
    exists, which is the evaluate track — including a project file written
    before this module knew about tracks.
    """
    for key, spec in TRACKS.items():
        if spec["goal"] == goal:
            return key
    return "evaluate"


def stages_for(goal: str) -> tuple[str, ...]:
    """The stages of this goal's track, in the order they are suggested in."""
    return TRACKS[track_of(goal)]["stages"]


def evaluate(goal: str, roles: set[str], has_prices: bool = False,
             has_geometry: bool = False, has_zone: bool = False,
             has_layers: bool = False, has_rate_column: bool = False) -> dict[str, Any]:
    """Check a goal against what the project actually holds.

    ``has_layers`` says whether anything at all is loaded: the trial design
    needs somewhere to put the strips, and any layer with points or polygons
    will give it an outline.

    ``has_rate_column`` says that a loaded file carries the rate beside the
    yield — a joined table, or a trial exported with both columns in it. The
    rate is then present whether or not a second file is marked as-applied,
    and asking for one would be asking for something the project already has.

    What comes back describes the goal, not a gate. Missing requirements are
    named so they can be fetched, never so a tab can be closed.
    """
    spec = GOALS.get(goal)
    if spec is None:
        raise ValueError(f"Unknown goal: '{goal}'.")

    def satisfied(key: str) -> bool:
        if key == "rate":
            return has_rate_column or bool(roles & {"as_applied", "plan"})
        if key == "prices":
            return has_prices
        if key == "geometry":
            return has_geometry or bool(roles & {"plan", "boundary"})
        if key == "field":
            return has_geometry or has_layers or bool(
                roles & {"boundary", "plan", "yield", "as_applied"}
            )
        if key == "zone":
            return has_zone
        return key in roles

    requirements = [
        Requirement(key, label, detail, satisfied(key))
        for key, label, detail in spec["requires"]
    ]
    optional = [
        Requirement(key, label, detail, satisfied(key))
        for key, label, detail in spec["optional"]
        if key not in OPTIONAL_WHEN_ROLE_PRESENT or key in roles
    ]
    missing = [r for r in requirements if not r.satisfied]
    track = track_of(goal)

    return {
        "goal": goal,
        "label": spec["label"],
        "description": spec["description"],
        "ready": not missing,
        "requirements": [r.to_dict() for r in requirements],
        "optional": [r.to_dict() for r in optional],
        "missing": [r.to_dict() for r in missing],
        # New keys, so callers that only read the old ones keep working.
        "track": track,
        "track_label": TRACKS[track]["label"],
        # What is missing keeps this goal from finishing; it keeps nothing
        # else from happening. Every tab stays open and every stage that has
        # what it needs still produces its own result.
        "blocks": False,
        "summary": (
            spec["description"] if not missing
            else "Still missing: " + ", ".join(r.label.lower() for r in missing) + "."
        ),
    }


def stage_of(
    layer_count: int,
    reviewed: bool,
    cleaned: bool,
    analysed: bool,
    exported: bool,
    goal: str = "difm",
    ready: bool = False,
    designed: bool = False,
) -> str:
    """Where the project currently stands.

    The stage is derived from what has happened, not stored, so it can never
    disagree with the actual state of the session.

    It says where the work has got to, not what is permitted: cleaning is
    passed over as soon as the data can be analysed as it stands (see the
    module docstring), and a stage that was skipped is still one click away.
    """
    if not layer_count:
        return "load"

    if track_of(goal) == "plan":
        return "export" if designed else "design"

    if not reviewed:
        return "review"
    if analysed:
        return "export"
    # Cleaning is suggested while it is still the thing most likely to
    # change the answer; once the analysis can run on what is loaded — or a
    # cleaning has already been run — the project is at the analysis.
    return "analyse" if (cleaned or ready) else "clean"


def stage_states(
    goal: str,
    stage: str,
    reviewed: bool = False,
    cleaned: bool = False,
    analysed: bool = False,
    designed: bool = False,
    exported: bool = False,
) -> list[dict[str, Any]]:
    """Every stage of this goal's track, with what became of it.

    ``state`` is one of ``done``, ``current``, ``skippable`` (it sits behind
    the current stage but nothing was produced there — the work went past it)
    and ``ahead``. Nothing is ever reported as blocked, and every stage
    carries ``reachable`` true: the tabs stay open whatever the state says.
    """
    stages = stages_for(goal)
    produced = {
        # A stage after 'load' is only ever reached with something open, so
        # by the time it is behind the current stage it is done, not skipped.
        "load": True,
        "review": reviewed,
        "clean": cleaned,
        "design": designed,
        "analyse": analysed,
        "export": exported,
    }
    try:
        current_index = stages.index(stage)
    except ValueError:
        current_index = 0

    out: list[dict[str, Any]] = []
    for index, key in enumerate(stages):
        if key == stage:
            state = "current"
        elif index < current_index:
            state = "done" if produced.get(key) else "skippable"
        else:
            state = "done" if produced.get(key) else "ahead"
        out.append({
            "key": key,
            "label": STAGE_LABELS[key],
            "description": STAGE_DESCRIPTIONS[key],
            "state": state,
            "reachable": True,
            **STAGE_OUTPUTS[key],
        })
    return out


def _stop_here(stage: str) -> dict[str, Any]:
    """The 'and if I stop here?' half of a next step."""
    output = STAGE_OUTPUTS[stage]
    return {
        "label": "You can stop here",
        "produces": output["produces"],
        "export": output["export"],
    }


def next_action(stage: str, goal: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    """The single next thing to do, why, and what stopping here would give.

    ``step`` and ``label`` are what they always were. ``stop`` is the other
    half of the answer: what this stage alone produces and how to take it
    away, because the next step is an offer and not an instruction.
    """
    def answer(step: str, label: str, why: str, at: str | None = None) -> dict[str, Any]:
        return {
            "step": step,
            "label": label,
            "why": why,
            "stop": _stop_here(at or stage),
            # No next step is a precondition for anything else: the tabs are
            # all open, and this is the one most likely to be worth doing.
            "blocks": False,
        }

    if stage == "load":
        return answer("load", "Load a file", "Nothing has been opened yet.")
    if stage == "review":
        return answer(
            "review", "Review the files that came in",
            "Units and roles have to be right before anything downstream "
            "inherits them.",
        )
    if not evaluation["ready"]:
        first = evaluation["missing"][0]
        return answer("load", f"Add: {first['label']}", first["detail"])
    if stage == "design":
        return answer(
            "design", "Lay the trial out over the field",
            "The strips and the AB line are what the monitor needs to drive the "
            "trial; everything after harvest is read against them.",
        )
    if stage == "clean":
        return answer(
            "clean", "Clean the data",
            "Raw monitor data carries overlap, turns and sensor faults that "
            "distort every statistic that follows.",
        )
    if stage == "analyse":
        return answer(
            "analyse", f"Run the {evaluation['label'].lower()}",
            evaluation["description"],
        )
    return answer(
        "export", "Build the package for the monitor",
        "The analysis is done; what is left is getting it onto the machine.",
    )


def describe() -> dict[str, Any]:
    """The whole workflow, for the interface to draw."""
    return {
        "stages": [
            {
                "key": key,
                "label": STAGE_LABELS[key],
                "description": STAGE_DESCRIPTIONS[key],
                **STAGE_OUTPUTS[key],
            }
            for key in STAGE_LABELS
        ],
        "tracks": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
                "goal": spec["goal"],
                "stages": list(spec["stages"]),
            }
            for key, spec in TRACKS.items()
        },
        "goals": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
                "track": track_of(key),
            }
            for key, spec in GOALS.items()
        },
    }
