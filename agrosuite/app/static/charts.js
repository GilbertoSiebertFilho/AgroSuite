/* Inline SVG charts.
 *
 * There are only a few, and they are specific — a comparative histogram, a
 * response curve, an elevation profile, a value across the height of the
 * field and an aspect rose — so drawing the SVG by hand comes out smaller
 * and more controllable than loading a whole charting library into an app
 * that has to open offline.
 *
 * Two rules hold across all of them. Nothing here converts a unit: every
 * number arrives in the unit already on screen, so a chart can never
 * disagree with the table beside it. And no chart draws two scales on one
 * plot — two measures that share no zero and no step are stacked on a
 * common x-axis instead, because where two lines on two scales cross is
 * decided by where the scales were put, not by the field. */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, parent = null) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* Overlaid before/after histogram: it shows where the cleaning took mass out
   * of the distribution, which is what says whether it cut a tail or the middle. */
  function histogram(container, before, after, options = {}) {
    container.innerHTML = "";
    const width = container.clientWidth || 360;
    const height = options.height || 150;
    const pad = { t: 8, r: 8, b: 20, l: 34 };

    if (!before?.counts?.length && !after?.counts?.length) {
      container.innerHTML = '<div class="empty">No data for the histogram.</div>';
      return;
    }

    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, height }, container);
    const series = [before, after].filter((s) => s?.counts?.length);
    const xMin = Math.min(...series.map((s) => s.edges[0]));
    const xMax = Math.max(...series.map((s) => s.edges[s.edges.length - 1]));
    const yMax = Math.max(...series.map((s) => Math.max(...s.counts)));
    if (!isFinite(xMin) || !isFinite(xMax) || xMax === xMin) {
      container.innerHTML = '<div class="empty">The distribution has no spread.</div>';
      return;
    }

    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * (width - pad.l - pad.r);
    const sy = (v) => height - pad.b - (v / yMax) * (height - pad.t - pad.b);

    const draw = (data, fill, opacity) => {
      if (!data?.counts?.length) return;
      for (let i = 0; i < data.counts.length; i++) {
        const x0 = sx(data.edges[i]);
        const x1 = sx(data.edges[i + 1]);
        const y = sy(data.counts[i]);
        el("rect", {
          x: x0, y, width: Math.max(0.6, x1 - x0 - 0.4),
          height: Math.max(0, height - pad.b - y), fill, opacity,
        }, svg);
      }
    };

    draw(before, css("--text-faint") || "#999", 0.5);
    draw(after, css("--accent") || "#2f7d4f", 0.75);

    el("line", {
      x1: pad.l, y1: height - pad.b, x2: width - pad.r, y2: height - pad.b,
      stroke: css("--border-strong"), "stroke-width": 1,
    }, svg);

    /* Breaks that mean something outside the data — the agronomic slope
     * classes — drawn where they fall, so the bars are read against them
     * rather than against the distribution's own shape. */
    for (const mark of options.marks || []) {
      if (!(mark.value > xMin && mark.value < xMax)) continue;
      const x = sx(mark.value);
      el("line", {
        x1: x, y1: pad.t, x2: x, y2: height - pad.b,
        stroke: css("--text-faint"), "stroke-width": 1, "stroke-dasharray": "2 3",
      }, svg);
      if (!mark.label) continue;
      const label = el("text", {
        x: x + 2, y: pad.t + 8, "font-size": 9, fill: css("--text-faint"),
      }, svg);
      label.textContent = mark.label;
    }

    const fmt = options.format || ((v) => Units.num(v, 0));
    for (const [value, anchor] of [[xMin, "start"], [xMax, "end"]]) {
      const t = el("text", {
        x: sx(value), y: height - 6, "text-anchor": anchor,
        "font-size": 10, fill: css("--text-faint"),
      }, svg);
      t.textContent = fmt(value);
    }
    const yLabel = el("text", {
      x: 4, y: pad.t + 8, "font-size": 10, fill: css("--text-faint"),
    }, svg);
    yLabel.textContent = Units.num(yMax, 0);
  }

  /* Yield curve and profit curve on the same rate axis, with the optimum
   * marked — the three things that decide the recommendation. */
  function responseCurve(container, curve, optimum, options = {}) {
    container.innerHTML = "";
    if (!curve?.length) {
      container.innerHTML = '<div class="empty">No fitted curve.</div>';
      return;
    }
    const width = container.clientWidth || 360;
    const height = options.height || 200;
    const pad = { t: 12, r: 44, b: 28, l: 46 };

    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, height }, container);

    // The curve arrives in internal units; the rate axis follows the chosen
    // input unit, or the chart contradicts the numbers beside it.
    const rateConv = options.rateConv || ((v) => v);
    const rates = curve.map((p) => rateConv(p.rate));
    const yields = curve.map((p) => options.yieldConv(p["yield"]));
    const profits = curve.map((p) => p.profit);

    const xMin = Math.min(...rates), xMax = Math.max(...rates);
    const yMin = Math.min(...yields), yMax = Math.max(...yields);
    const pMin = Math.min(...profits), pMax = Math.max(...profits);

    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * (width - pad.l - pad.r);
    const syY = (v) => height - pad.b - ((v - yMin) / (yMax - yMin || 1)) * (height - pad.t - pad.b);
    const syP = (v) => height - pad.b - ((v - pMin) / (pMax - pMin || 1)) * (height - pad.t - pad.b);

    el("line", {
      x1: pad.l, y1: height - pad.b, x2: width - pad.r, y2: height - pad.b,
      stroke: css("--border-strong"),
    }, svg);

    const path = (values, scale, stroke, dash) => {
      const d = values.map((v, i) => `${i ? "L" : "M"}${sx(rates[i]).toFixed(1)},${scale(v).toFixed(1)}`).join(" ");
      el("path", {
        d, fill: "none", stroke, "stroke-width": 2,
        "stroke-dasharray": dash || "none", "stroke-linejoin": "round",
      }, svg);
    };

    path(yields, syY, css("--accent"));
    path(profits, syP, css("--warn"), "4 3");

    if (optimum?.optimum_rate != null) {
      const x = sx(rateConv(optimum.optimum_rate));
      el("line", {
        x1: x, y1: pad.t, x2: x, y2: height - pad.b,
        stroke: css("--text-muted"), "stroke-width": 1, "stroke-dasharray": "2 3",
      }, svg);
      const t = el("text", {
        x: Math.min(x + 4, width - pad.r - 4), y: pad.t + 9,
        "font-size": 10, fill: css("--text-muted"),
      }, svg);
      t.textContent = `optimum ${Units.num(rateConv(optimum.optimum_rate), 0)}`;
    }

    const tick = (x, y, text, anchor, color) => {
      const t = el("text", { x, y, "text-anchor": anchor, "font-size": 10, fill: color }, svg);
      t.textContent = text;
    };
    tick(pad.l, height - 8, Units.num(xMin, 0), "start", css("--text-faint"));
    tick(width - pad.r, height - 8, Units.num(xMax, 0), "end", css("--text-faint"));
    tick(4, pad.t + 8, Units.num(yMax, 0), "start", css("--accent"));
    tick(4, height - pad.b, Units.num(yMin, 0), "start", css("--accent"));
    tick(width - 4, pad.t + 8, Units.num(pMax, 0), "end", css("--warn"));
    tick(width - 4, height - pad.b, Units.num(pMin, 0), "end", css("--warn"));

    const legend = el("text", {
      x: pad.l, y: pad.t + 9, "font-size": 10, fill: css("--text-muted"),
    }, svg);
    legend.textContent =
      `— yield (${options.yieldUnit})   ╌ profit   ·   rate in ${options.rateUnit || ""}`;
  }

  /* Elevation along a line drawn on the map — and, when a value layer is
   * picked, that value along the same line under it.
   *
   * Two panels stacked on one distance axis, never two y-scales on one
   * plot. A height in feet and a yield in bushels share no zero and no
   * step, so where two lines drawn against two scales cross is decided by
   * where the scales were put, not by the field: the chart would show a
   * relation the data never claimed. Stacked, the same two series are read
   * against the same distance and the reader compares shapes, which is the
   * comparison that is actually there.
   *
   * Stations that fall outside the field come back null, and the path
   * breaks there rather than joining across the gap — a straight segment
   * over ground that was never measured is an invention, and it is exactly
   * where a line clipping the corner of a field would draw one. The value
   * series breaks the same way, wherever no reading fell near the line. */
  function profile(container, series, options = {}) {
    container.innerHTML = "";
    const distance = series?.distance || [];
    const elevation = series?.elevation || [];
    const values = series?.values || null;
    const points = distance
      .map((d, i) => ({ d, z: elevation[i], v: values ? values[i] : null }))
      .filter((p) => Number.isFinite(p.d));
    const measured = points.filter((p) => Number.isFinite(p.z));
    if (measured.length < 2) {
      container.innerHTML = '<div class="empty">The line crosses too little of the field to draw.</div>';
      return;
    }
    // One panel is worth drawing only when two stations of it were
    // measured; a single point is a dot nobody can read a shape off.
    const twoPanels = points.filter((p) => Number.isFinite(p.v)).length >= 2;

    const width = container.clientWidth || 360;
    const legendHeight = twoPanels ? 16 : 0;
    const height = options.height || (twoPanels ? 258 : 170);
    const pad = { t: 14 + legendHeight, r: 10, b: 24, l: 44 };
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, height }, container);

    const xMin = 0, xMax = Math.max(...points.map((p) => p.d)) || 1;
    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * (width - pad.l - pad.r);

    const available = height - pad.t - pad.b;
    const gap = 20;
    const heightPanel = twoPanels ? (available - gap) * 0.55 : available;
    const panels = [{
      get: (p) => p.z, top: pad.t, bottom: pad.t + heightPanel,
      color: css("--accent"), unit: options.elevationUnit || "",
      label: options.elevationLabel || "Height", decimals: 1,
    }];
    if (twoPanels) {
      panels.push({
        get: (p) => p.v, top: pad.t + heightPanel + gap, bottom: height - pad.b,
        color: css("--series-value"), unit: options.valueUnit || "",
        label: options.valueLabel || "Value", decimals: 1,
      });
    }

    for (const panel of panels) {
      const seen = points.map(panel.get).filter((v) => Number.isFinite(v));
      let low = Math.min(...seen);
      let top = Math.max(...seen);
      // A profile along a contour is flat, not an error: a band around the
      // height keeps the line in the middle instead of magnifying millimetres.
      if (top - low < 1e-6) { low -= 0.5; top += 0.5; }
      const span = top - low;
      const sy = (v) => panel.bottom - ((v - low) / span) * (panel.bottom - panel.top);

      el("line", {
        x1: pad.l, y1: panel.bottom, x2: width - pad.r, y2: panel.bottom,
        stroke: css("--border-strong"), "stroke-width": 1,
      }, svg);

      // One path per measured run, so a gap stays a gap.
      let run = [];
      const flush = () => {
        if (run.length >= 2) {
          el("path", {
            d: run.map((p, i) =>
              `${i ? "L" : "M"}${sx(p.d).toFixed(1)},${sy(panel.get(p)).toFixed(1)}`).join(" "),
            fill: "none", stroke: panel.color, "stroke-width": 2,
            "stroke-linejoin": "round", "stroke-linecap": "round",
          }, svg);
        } else if (run.length === 1) {
          el("circle", {
            cx: sx(run[0].d), cy: sy(panel.get(run[0])), r: 2, fill: panel.color,
          }, svg);
        }
        run = [];
      };
      for (const point of points) {
        if (Number.isFinite(panel.get(point))) run.push(point);
        else flush();
      }
      flush();

      // The unit rides on the top tick of each axis rather than on a
      // floating label, which on a line that ends high would sit over the
      // ground it describes. The ticks are ink, not series colour: the
      // line key in the legend carries which series they belong to.
      const unit = panel.unit ? ` ${panel.unit}` : "";
      tick(svg, 4, panel.top + 4, `${Units.num(top, panel.decimals)}${unit}`, "start",
           css("--text-faint"));
      tick(svg, 4, panel.bottom, Units.num(low, panel.decimals), "start", css("--text-faint"));
    }

    const distanceUnit = options.distanceUnit ? ` ${options.distanceUnit}` : "";
    tick(svg, pad.l, height - 7, "0", "start", css("--text-faint"));
    tick(svg, width - pad.r, height - 7, `${Units.num(xMax, 0)}${distanceUnit}`, "end",
         css("--text-faint"));

    if (twoPanels) {
      legend(svg, pad.l, 10, panels.map((panel) => ({
        color: panel.color,
        text: panel.unit ? `${panel.label} (${panel.unit})` : panel.label,
      })));
    }

    // One hit target per station, the full height of the chart and at least
    // as wide as the gap to its neighbour, so the pointer only has to be
    // over the right distance rather than on a 2 px line. It reads out
    // every panel at that distance at once — the reader should never have
    // to land on a particular series to get its number — and each of those
    // numbers is in the station table under the chart as well.
    const step = (width - pad.l - pad.r) / Math.max(points.length - 1, 1);
    for (const point of points) {
      const hit = el("rect", {
        x: sx(point.d) - step / 2, y: pad.t, width: Math.max(step, 2),
        height: height - pad.b - pad.t, fill: "transparent",
      }, svg);
      const lines = [`${Units.num(point.d, 0)}${distanceUnit} along the line`];
      for (const panel of panels) {
        const value = panel.get(point);
        lines.push(`${panel.label}: ${Number.isFinite(value)
          ? `${Units.num(value, panel.decimals)}${panel.unit ? ` ${panel.unit}` : ""}`
          : "nothing measured here"}`);
      }
      el("title", {}, hit).textContent = lines.join("\n");
    }
  }

  /* A text tick. Every number on a chart wears an ink colour, never its
   * series' — a light hue is unreadable as text, and identity comes from
   * the coloured key beside the label instead. */
  function tick(svg, x, y, text, anchor, color) {
    const node = el("text", { x, y, "text-anchor": anchor, "font-size": 10, fill: color }, svg);
    node.textContent = text;
    return node;
  }

  /* A row of short coloured strokes with their names: the dependable way
   * to tell two series apart, since colour alone fails a reader who cannot
   * separate the two hues. Mirrors the mark — a line for a line. */
  function legend(svg, x, y, entries) {
    let cursor = x;
    for (const entry of entries) {
      el("line", {
        x1: cursor, y1: y - 3, x2: cursor + 12, y2: y - 3,
        stroke: entry.color, "stroke-width": 2, "stroke-linecap": "round",
      }, svg);
      const label = tick(svg, cursor + 16, y, entry.text, "start", css("--text-muted"));
      cursor += 16 + entry.text.length * 5.6 + 14;
      label.setAttribute("font-size", 10);
    }
  }

  /* The value against the height of the ground: the chart the relief
   * raises the question for.
   *
   * One measure, one hue. The mean of each elevation band is a line across
   * the height of the field, the quartiles behind it as a wash, and the
   * field average as a rule so that "above or below average" is read
   * without arithmetic. A second, thinner panel under it, on the same
   * height axis, shows how much ground the field actually has at each
   * height — the band at the top of a hill may be a hectare, and a mean
   * drawn from a hectare is a thinner claim than one drawn from ten.
   *
   * The two panels share the x-axis rather than sharing a plot: the value
   * and the hectares have nothing in common but the height they are read
   * at, and stacking them says exactly that.
   *
   * Everything arrives converted — the caller has already put the value in
   * the unit on screen and the heights in feet or metres — because a chart
   * that converted its own numbers would disagree with the table beside
   * it the first time one of them was changed. */
  function valueAcrossElevation(container, data, options = {}) {
    container.innerHTML = "";
    const bands = (data?.bands || []).filter(
      (b) => Number.isFinite(b.at) && Number.isFinite(b.mean));
    if (bands.length < 2) {
      container.innerHTML =
        '<div class="empty">Too few bands to draw a shape across the field.</div>';
      return;
    }
    const density = data?.density?.counts?.length ? data.density : null;

    const width = container.clientWidth || 360;
    const height = options.height || 250;
    const strip = 30;
    const gap = 16;
    const pad = { t: 16, r: 12, b: 24, l: 48 };
    const plotBottom = height - pad.b - (density ? strip + gap : 0);
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, height }, container);

    const edges = bands.flatMap((b) => [b.from, b.to]).filter(Number.isFinite);
    const xMin = Math.min(...edges, ...(density ? [density.edges[0]] : []));
    const xMax = Math.max(...edges, ...(density ? [density.edges[density.edges.length - 1]] : []));
    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * (width - pad.l - pad.r);

    const spread = bands.flatMap((b) => [b.p25, b.p75, b.mean]).filter(Number.isFinite);
    const fieldMean = Number.isFinite(data?.fieldMean) ? data.fieldMean : null;
    if (fieldMean != null) spread.push(fieldMean);
    let yMin = Math.min(...spread);
    let yMax = Math.max(...spread);
    // A little air above and below, so the top of the quartile band is not
    // painted onto the frame and the reader can see it ends.
    const margin = (yMax - yMin || Math.abs(yMax) || 1) * 0.08;
    yMin -= margin;
    yMax += margin;
    const sy = (v) => plotBottom - ((v - yMin) / (yMax - yMin || 1)) * (plotBottom - pad.t);

    // The quartiles behind the line: a wash of the series hue, never a
    // block — it is context for the mean, not a second series.
    const quartiles = bands.filter((b) => Number.isFinite(b.p25) && Number.isFinite(b.p75));
    if (quartiles.length >= 2) {
      const upper = quartiles.map((b) => `${sx(b.at).toFixed(1)},${sy(b.p75).toFixed(1)}`);
      const lower = quartiles.slice().reverse()
        .map((b) => `${sx(b.at).toFixed(1)},${sy(b.p25).toFixed(1)}`);
      el("path", {
        d: `M${upper.join("L")}L${lower.join("L")}Z`,
        fill: css("--series-value"), opacity: 0.12, stroke: "none",
      }, svg);
    }

    // The field average, so "above or below" needs no arithmetic. A solid
    // hairline in ink: it is a reference the reader looks for, not a grid.
    if (fieldMean != null) {
      const y = sy(fieldMean);
      el("line", {
        x1: pad.l, y1: y, x2: width - pad.r, y2: y,
        stroke: css("--text-faint"), "stroke-width": 1,
      }, svg);
      const note = tick(svg, width - pad.r, y - 4, "field average", "end", css("--text-muted"));
      note.setAttribute("font-size", 9);
    }

    el("line", {
      x1: pad.l, y1: plotBottom, x2: width - pad.r, y2: plotBottom,
      stroke: css("--border-strong"), "stroke-width": 1,
    }, svg);

    el("path", {
      d: bands.map((b, i) =>
        `${i ? "L" : "M"}${sx(b.at).toFixed(1)},${sy(b.mean).toFixed(1)}`).join(" "),
      fill: "none", stroke: css("--series-value"), "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }, svg);
    for (const band of bands) {
      // The surface ring keeps a marker legible where it crosses the wash
      // or the average line, and it is part of what the pointer can hit.
      el("circle", {
        cx: sx(band.at), cy: sy(band.mean), r: 4,
        fill: css("--series-value"), stroke: css("--surface"), "stroke-width": 2,
      }, svg);
    }

    // How much ground lies at each height, on the same axis: a band at the
    // extreme that carries a hectare is a thinner claim than one carrying
    // ten, and the strip is what shows it without a second chart.
    if (density) {
      const top = plotBottom + gap;
      const most = Math.max(...density.counts) || 1;
      for (let i = 0; i < density.counts.length; i++) {
        const x0 = sx(density.edges[i]);
        const x1 = sx(density.edges[i + 1]);
        const barHeight = (density.counts[i] / most) * strip;
        el("rect", {
          x: x0, y: top + strip - barHeight,
          width: Math.max(0.6, x1 - x0 - 2), height: Math.max(0, barHeight),
          fill: css("--text-faint"), opacity: 0.45,
        }, svg);
      }
      el("line", {
        x1: pad.l, y1: top + strip, x2: width - pad.r, y2: top + strip,
        stroke: css("--border-strong"), "stroke-width": 1,
      }, svg);
      const caption = tick(svg, 4, top + 8, options.densityLabel || "ground", "start",
                           css("--text-faint"));
      caption.setAttribute("font-size", 9);
    }

    const valueUnit = options.valueUnit ? ` ${options.valueUnit}` : "";
    const lengthUnit = options.lengthUnit ? ` ${options.lengthUnit}` : "";
    tick(svg, 4, pad.t + 4, `${Units.num(yMax, 0)}${valueUnit}`, "start", css("--text-faint"));
    tick(svg, 4, plotBottom, Units.num(yMin, 0), "start", css("--text-faint"));
    tick(svg, pad.l, height - 7, Units.num(xMin, 0), "start", css("--text-faint"));
    tick(svg, width - pad.r, height - 7, `${Units.num(xMax, 0)}${lengthUnit}`, "end",
         css("--text-faint"));

    // The hit target is the whole column of the band, top to bottom, so
    // the pointer only has to be over the right height rather than on the
    // 2 px line. Every number it shows is in the table under the chart
    // too: the hover adds, it never gates.
    for (const band of bands) {
      const x0 = sx(Number.isFinite(band.from) ? band.from : band.at);
      const x1 = sx(Number.isFinite(band.to) ? band.to : band.at);
      const hit = el("rect", {
        x: Math.min(x0, x1), y: pad.t, width: Math.max(2, Math.abs(x1 - x0)),
        height: height - pad.b - pad.t, fill: "transparent",
      }, svg);
      el("title", {}, hit).textContent = band.tooltip || "";
    }
  }

  /* The aspect rose: which way the field faces, as eight petals whose
   * length is the share of the field facing that way. Ground too flat to
   * have an aspect is named under the rose rather than drawn as a ninth
   * petal — it points nowhere, and a petal would give it a direction. */
  function rose(container, sectors, options = {}) {
    container.innerHTML = "";
    const bearings = {
      N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315,
    };
    const petals = (sectors || []).filter((s) => s.key in bearings);
    if (!petals.length) {
      container.innerHTML = '<div class="empty">No aspect to show.</div>';
      return;
    }
    const size = options.size || Math.min(container.clientWidth || 220, 240);
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${size} ${size}`, height: size }, container);
    const cx = size / 2, cy = size / 2;
    const radius = size / 2 - 16;
    const max = Math.max(...petals.map((p) => p.pct || 0)) || 1;

    for (const fraction of [0.5, 1]) {
      el("circle", {
        cx, cy, r: radius * fraction, fill: "none",
        stroke: css("--border"), "stroke-width": 1, "stroke-dasharray": "2 3",
      }, svg);
    }

    const point = (bearing, r) => [
      cx + r * Math.sin((bearing * Math.PI) / 180),
      cy - r * Math.cos((bearing * Math.PI) / 180),
    ];
    for (const petal of petals) {
      const r = radius * ((petal.pct || 0) / max);
      if (r <= 0.5) continue;
      const centre = bearings[petal.key];
      const [x1, y1] = point(centre - 22.5, r);
      const [x2, y2] = point(centre + 22.5, r);
      const wedge = el("path", {
        d: `M${cx},${cy} L${x1.toFixed(1)},${y1.toFixed(1)} A${r.toFixed(1)},${r.toFixed(1)} 0 0 1 ` +
           `${x2.toFixed(1)},${y2.toFixed(1)} Z`,
        fill: css("--accent"), opacity: 0.75, stroke: css("--surface"), "stroke-width": 1,
      }, svg);
      el("title", {}, wedge).textContent = `${petal.label}: ${Units.num(petal.pct, 1)}%`;
    }

    for (const [key, bearing] of Object.entries(bearings)) {
      const [x, y] = point(bearing, radius + 9);
      const label = el("text", {
        x, y: y + 3, "text-anchor": "middle", "font-size": 9,
        fill: key === "N" ? css("--text") : css("--text-faint"),
      }, svg);
      label.textContent = key;
    }
    const scale = el("text", {
      x: cx + 3, y: cy - radius * 0.5 - 2, "font-size": 9, fill: css("--text-faint"),
    }, svg);
    scale.textContent = `${Units.num(max / 2, 0)}%`;
  }

  /* Horizontal bars for the removal counts by reason. */
  function bars(container, rows, options = {}) {
    container.innerHTML = "";
    if (!rows?.length) {
      container.innerHTML = '<div class="empty">No record was removed.</div>';
      return;
    }
    const max = Math.max(...rows.map((r) => r.value));
    const list = document.createElement("div");
    for (const row of rows) {
      const wrap = document.createElement("div");
      wrap.style.marginBottom = "7px";
      const top = document.createElement("div");
      top.style.cssText = "display:flex;justify-content:space-between;font-size:12px;gap:8px";
      top.innerHTML =
        `<span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${row.label}</span>` +
        `<span style="font-family:var(--mono);color:var(--text-muted);flex:none">${Units.num(row.value, 0)}${
          options.suffix ? " " + options.suffix : ""}</span>`;
      const track = document.createElement("div");
      track.className = "progress-track";
      track.style.marginTop = "3px";
      const fill = document.createElement("div");
      fill.className = "progress-fill";
      fill.style.width = `${(row.value / max) * 100}%`;
      track.appendChild(fill);
      wrap.append(top, track);
      list.appendChild(wrap);
    }
    container.appendChild(list);
  }

  return { histogram, responseCurve, profile, valueAcrossElevation, rose, bars };
})();
