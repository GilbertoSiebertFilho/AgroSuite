# Driving AgroSuite from Claude

AgroSuite ships an MCP server, so Claude can do the work you would otherwise
click through. The point is not to replace the interface — a map and a
cleaning report are things to look at — but to let you ask for the work in one
sentence instead of five screens.

> Load the three files in `C:\Data\NW-14-32-W2`, they're in bu/ac and lb/ac
> with feet and mph. Clean the yield map, join the layers and tell me the
> optimum nitrogen rate for canola at $16.50 a bushel and $0.62 a pound of N.

## Setting it up

### Claude Desktop

Open **Settings → Developer → Edit Config** and add the server:

```json
{
  "mcpServers": {
    "agrosuite": {
      "command": "C:\\path\\to\\AgroSuite\\.venv\\Scripts\\python.exe",
      "args": ["-m", "agrosuite.mcp_server"],
      "cwd": "C:\\path\\to\\AgroSuite"
    }
  }
}
```

On Linux or macOS the command is `.venv/bin/python` and the paths use forward
slashes. Restart Claude Desktop; AgroSuite appears in the tools list.

### Claude Code

```
claude mcp add agrosuite -- /path/to/AgroSuite/.venv/bin/python -m agrosuite.mcp_server
```

## How it behaves

The server starts AgroSuite itself if it is not already running, headless —
no browser window opens unasked. If the app is already up on port 8765 (or
the next free one), it attaches to that instead, so what Claude does and what
you see in the window are the same session.

Every tool maps to an endpoint the interface itself uses. The MCP surface can
do no more than a person clicking could, and nothing writes outside the app's
own workspace.

## The tools

| Tool | What it does |
|---|---|
| `open_file` | Opens a monitor file and reports the preliminary analysis |
| `list_datasets`, `describe_dataset` | What is loaded, and one dataset's detail |
| `project_status` | Stage, roles, what is missing, what to do next |
| `set_role`, `set_prices` | Say what a file is for; set crop price and input cost |
| `clean_dataset` | Runs the cleaning and returns the report |
| `join_layers` | Puts plan, as-applied and yield on a shared grid |
| `analyse_economics` | Fits the yield response to rate and finds the economic optimum |
| `analyse_terrain` | Reads the relief of a field and describes it in sentences |
| `terrain_zones` | Turns that relief into a zone layer the export tools take |
| `design_trial` | Lays out a randomized block strip trial |
| `list_usb_drives`, `plan_usb_write`, `write_to_usb` | Copying to a stick |
| `validate_package` | Checks a package against the known causes of rejection |

## Asking about the relief

The altitude a monitor logs on every point is enough to read the relief of
the field, and `analyse_terrain` answers in sentences rather than tables:

> Analyse the relief of the yield map and tell me where water is going to sit.

It reports the character of the field, how much it falls and in which
direction, the hills and the low ground with where they are and how big they
are, the closed depressions with the volume each one holds, the share above
10 % slope and the likely wet ground — each number with its unit. Then
`terrain_zones` cuts that relief into zones by landform, slope class,
elevation band or wetness, and hands back an ordinary dataset id, which is
what the export tools take. [terrain.md](terrain.md) has the whole surface.

## Units, when asking

The tools take the app's internal units: **kg/ha** for rates and yields,
**metres** for distances and every height, depth and cell size the terrain
tools take, **price per kilogram**. Claude converts on the way in, but it
helps to state what your numbers are in.

What comes back out of `analyse_terrain` is the other way round: it is
written in the unit set the app is showing — the picker at the top of the
window — and its first line says which. That is so the sentence Claude
reads back to you is the sentence on your screen, word for word, rather
than the same field described in two systems at once.

Prices convert like this:

- dollars per bushel ÷ the crop's bushel weight in kg
  (canola 22.68, wheat 27.22, barley 21.77, oats 14.52, corn 25.40)
- dollars per pound ÷ 0.4536

So $16.50/bu of canola is $0.7276/kg, and $0.62/lb of N is $1.3668/kg.

## Two things it will not do

**Overwrite a USB stick without being told.** `write_to_usb` leaves alone
anything already on the drive unless its name is passed explicitly. Ask for
`plan_usb_write` first: it reports what would be replaced and writes nothing.

**Claim a file will work on your monitor.** `validate_package` rules out the
known causes of rejection — it cannot test your display's firmware. The final
check is still loading the package on the machine before heading out.
