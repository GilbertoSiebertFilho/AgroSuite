/* Map and point layer.
 *
 * The points do not become Leaflet markers: a yield map has tens of thousands
 * of them, and one DOM element per point locks up the browser. Instead a canvas
 * laid over the map is redrawn on every move, projecting the coordinates with
 * Leaflet's own maths.
 *
 * It is a factory rather than a singleton because the cleaning tab shows two
 * maps side by side — the field before and after — and each needs its own
 * canvas, hover index and tooltip. `MapView` at the bottom is the main map;
 * a second pane is `createMapView("map-b", "point-canvas-b")`. */

function createMapView(defaultMapId, defaultCanvasId) {
  let map = null;
  let canvas = null;
  let ctx = null;
  let basemap = null;
  let overlayGroup = null;
  let data = null;      // { lon: [], lat: [], values: [], scale: {} }
  let overlay = null;   // { lon: [], lat: [], groups: [], palette: {} }
  let colorScale = null;
  let unitLabel = "";   // what the tooltip writes after the number
  let pointSize = 3;
  let imageLayer = null;        // one raster overlay at a time (the relief)
  let imageUrl = "";            // what it is showing, so a redraw is not a refetch
  let imageFailed = null;       // what to call when the server will not serve it
  const named = new Map();      // vector layers that toggle independently
  let clickThrough = false;     // a tool is drawing: the layers must not eat the clicks

  /* Its own pane, between the imagery and the vectors: see setImage. */
  const IMAGE_PANE = "rasterOverlay";

  /* Hover: the tooltip finds the point under the cursor through a grid of
   * screen-pixel cells rather than a scan of every point — at 60,000 points a
   * scan per mouse move would stutter. The grid is rebuilt lazily, only after
   * the view settles, because every move invalidates every projection. */
  const HIT_RADIUS_PX = 8;
  let index = null;
  let indexDirty = true;
  let moving = false;
  let tooltip = null;
  let hoverPoint = null;
  let hoverFrame = 0;
  let resizeObserver = null;

  /* A cool-to-warm ramp, legible in both light and dark themes and readable by
   * people who do not see green and red as separate colours: luminance rises
   * monotonically from the minimum to the maximum. */
  const RAMP = [
    [ 49,  54, 149],
    [ 69, 117, 180],
    [116, 173, 209],
    [171, 217, 233],
    [254, 224, 144],
    [253, 174,  97],
    [244, 109,  67],
    [215,  48,  39],
  ];

  function rampColor(t) {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const a = RAMP[i], b = RAMP[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }

  function rampCss() {
    const stops = RAMP.map((c, i) =>
      `rgb(${c.join(",")}) ${((i / (RAMP.length - 1)) * 100).toFixed(0)}%`);
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }

  function onWindowResize() { resize(); draw(); indexDirty = true; }

  function init(mapId = defaultMapId, canvasId = defaultCanvasId) {
    map = L.map(mapId, { preferCanvas: true, zoomControl: true, attributionControl: false })
      .setView([-23.5, -51.2], 15);

    basemap = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 21, maxNativeZoom: 19 },
    );
    basemap.addTo(map);
    overlayGroup = L.layerGroup().addTo(map);

    canvas = document.getElementById(canvasId);
    ctx = canvas.getContext("2d");

    map.on("move zoom viewreset zoomend moveend", draw);
    // Leaflet fires 'resize' from invalidateSize: the canvas has to follow the
    // container, or the points drift off the imagery when a pane is added.
    map.on("resize", () => { resize(); draw(); indexDirty = true; });
    map.on("movestart zoomstart", () => { moving = true; hideTooltip(); });
    map.on("moveend zoomend", () => { moving = false; indexDirty = true; scheduleHover(); });
    window.addEventListener("resize", onWindowResize);
    // The container changes size without the window doing so — the stage
    // strip appears once a project exists, a second pane comes and goes — and
    // Leaflet only re-measures on request. Left stale, its centre is off by
    // half the difference and every point lands beside its imagery.
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => { if (map) invalidateSize(); });
      resizeObserver.observe(map.getContainer());
    }
    resize();
    bindHover();
    return map;
  }

  /* Release the Leaflet instance and the listeners this view added outside
   * its container, so a pane can be created and removed as often as the
   * user toggles the comparison without leaking a map each time. */
  function destroy() {
    if (!map) return;
    window.removeEventListener("resize", onWindowResize);
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (hoverFrame) cancelAnimationFrame(hoverFrame);
    hoverFrame = 0;
    clearNamed();
    clearImage();
    map.remove();
    map = null; canvas = null; ctx = null; basemap = null; overlayGroup = null;
    data = null; overlay = null; colorScale = null; index = null; tooltip = null;
  }

  function setBasemap(on) {
    if (!basemap || !map) return;
    if (on && !map.hasLayer(basemap)) basemap.addTo(map);
    if (!on && map.hasLayer(basemap)) map.removeLayer(basemap);
  }

  function resize() {
    if (!canvas || !map) return;
    const size = map.getSize();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = size.x * ratio;
    canvas.height = size.y * ratio;
    canvas.style.width = `${size.x}px`;
    canvas.style.height = `${size.y}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  /* After the layout around the map changes — a second pane appears or goes
   * away — Leaflet has to re-measure its container, and the canvas with it. */
  function invalidateSize() {
    if (!map) return;
    map.invalidateSize({ animate: false });
    resize();
    draw();
    indexDirty = true;
  }

  /* Feed the point layer. `convert` translates the internal value into the
   * chosen unit — the colour scale is computed in the displayed unit, so the
   * legend and the colours tell the same story.
   *
   * `options.unit` is the label the tooltip writes after the number, and
   * `options.scale` ({ low, high }, in the displayed unit) pins the colour
   * scale instead of deriving it from this payload: two maps compared side by
   * side must share one scale, or the eye reads a rescaling as a change. */
  function setPoints(payload, convert, options = {}) {
    data = payload;
    const fixed = options.scale;
    const scale = payload?.scale;
    if (fixed && Number.isFinite(fixed.low) && Number.isFinite(fixed.high)) {
      colorScale = { low: fixed.low, high: fixed.high, span: (fixed.high - fixed.low) || 1 };
    } else if (scale) {
      const low = convert ? convert(scale.low) : scale.low;
      const high = convert ? convert(scale.high) : scale.high;
      colorScale = { low, high, span: (high - low) || 1 };
    } else {
      colorScale = null;
    }
    unitLabel = options.unit ?? "";
    if (data) data._convert = convert || ((v) => v);
    indexDirty = true;
    resize();
    draw();
    scheduleHover();
    return colorScale;
  }

  function clearPoints() {
    data = null; colorScale = null; indexDirty = true;
    hideTooltip(); draw();
  }

  /* A second set of points drawn over the first, coloured by a category
   * rather than by a value. It is how the app shows *why* each record was
   * removed, which is the thing that makes a cleaning judgeable: 10% removed
   * is meaningless until you can see that it was the headland and the
   * overlap, and not a strip through the middle of the field. */
  function setOverlay(payload, palette) {
    overlay = payload ? { ...payload, palette: palette || {} } : null;
    indexDirty = true;
    draw();
    scheduleHover();
  }

  function clearOverlay() { overlay = null; indexDirty = true; draw(); scheduleHover(); }

  function draw() {
    if (!ctx || !canvas || !map) return;
    const size = map.getSize();
    ctx.clearRect(0, 0, size.x, size.y);
    if (!data || !data.lon?.length) return;

    const bounds = map.getBounds().pad(0.05);
    const west = bounds.getWest(), east = bounds.getEast();
    const south = bounds.getSouth(), north = bounds.getNorth();

    const zoom = map.getZoom();
    const radius = Math.max(1.4, Math.min(7, pointSize * (zoom / 15)));
    const hasValues = data.values?.length === data.lon.length;
    const convert = data._convert || ((v) => v);

    const neutral = getComputedStyle(document.documentElement)
      .getPropertyValue("--accent").trim() || "#2f7d4f";

    ctx.globalAlpha = 0.85;
    for (let i = 0; i < data.lon.length; i++) {
      const lon = data.lon[i], lat = data.lat[i];
      if (lon < west || lon > east || lat < south || lat > north) continue;

      const p = map.latLngToContainerPoint([lat, lon]);
      if (hasValues && colorScale) {
        const raw = data.values[i];
        if (raw == null) continue;
        const t = (convert(raw) - colorScale.low) / colorScale.span;
        const c = rampColor(t);
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      } else {
        ctx.fillStyle = neutral;
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, 6.283185307179586);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (overlay?.lon?.length) drawOverlay(bounds, zoom);
  }

  function drawOverlay(bounds, zoom) {
    const west = bounds.getWest(), east = bounds.getEast();
    const south = bounds.getSouth(), north = bounds.getNorth();
    const radius = Math.max(1.6, Math.min(8, 3.4 * (zoom / 15)));
    const fallback = "#888";

    ctx.globalAlpha = 0.95;
    for (let i = 0; i < overlay.lon.length; i++) {
      const lon = overlay.lon[i], lat = overlay.lat[i];
      if (lon < west || lon > east || lat < south || lat > north) continue;
      const group = overlay.groups?.[i];
      ctx.fillStyle = overlay.palette[group] || fallback;
      const p = map.latLngToContainerPoint([lat, lon]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, 6.283185307179586);
      ctx.fill();
      // A thin ring separates a removed point from the kept ones underneath.
      ctx.strokeStyle = "rgba(255,255,255,.7)";
      ctx.lineWidth = 0.7;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /* ------------------------------------------------------------- hover --- */

  function bindHover() {
    const container = map.getContainer();
    tooltip = document.createElement("div");
    tooltip.className = "map-tooltip";
    tooltip.hidden = true;
    tooltip.innerHTML = '<span class="v"></span> <span class="u"></span>';
    container.appendChild(tooltip);

    container.addEventListener("mousemove", (event) => {
      hoverPoint = map.mouseEventToContainerPoint(event);
      scheduleHover();
    });
    container.addEventListener("mouseleave", () => { hoverPoint = null; hideTooltip(); });
  }

  /* One lookup per frame at most, however fast the mouse moves. */
  function scheduleHover() {
    if (!hoverPoint || hoverFrame || !map) return;
    hoverFrame = requestAnimationFrame(updateTooltip);
  }

  function hideTooltip() { if (tooltip) tooltip.hidden = true; }

  function cellKey(x, y) {
    return Math.floor(x / HIT_RADIUS_PX) * 65536 + Math.floor(y / HIT_RADIUS_PX);
  }

  /* Cells are as wide as the hit radius, so a query touches at most the 3×3
   * block around the cursor. Entries are flat quadruples (x, y, kind, i)
   * rather than objects: tens of thousands of small objects per rebuild is
   * the kind of garbage that makes a pan feel sticky. */
  function buildIndex() {
    index = new Map();
    indexDirty = false;
    if (!map) return;
    const size = map.getSize();
    const bounds = map.getBounds();
    const west = bounds.getWest(), east = bounds.getEast();
    const south = bounds.getSouth(), north = bounds.getNorth();

    const add = (lon, lat, kind, i) => {
      if (lon < west || lon > east || lat < south || lat > north) return;
      const p = map.latLngToContainerPoint([lat, lon]);
      if (p.x < 0 || p.y < 0 || p.x > size.x || p.y > size.y) return;
      const key = cellKey(p.x, p.y);
      let cell = index.get(key);
      if (!cell) { cell = []; index.set(key, cell); }
      cell.push(p.x, p.y, kind, i);
    };

    if (data?.lon?.length) {
      for (let i = 0; i < data.lon.length; i++) add(data.lon[i], data.lat[i], 0, i);
    }
    if (overlay?.lon?.length) {
      for (let i = 0; i < overlay.lon.length; i++) add(overlay.lon[i], overlay.lat[i], 1, i);
    }
  }

  function nearest(x, y) {
    if (indexDirty) buildIndex();
    if (!index?.size) return null;
    const cx = Math.floor(x / HIT_RADIUS_PX), cy = Math.floor(y / HIT_RADIUS_PX);
    const limit = HIT_RADIUS_PX * HIT_RADIUS_PX;
    const tie = 0.25;   // a quarter of a pixel squared: the same record twice
    let best = null;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = index.get((cx + dx) * 65536 + (cy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k += 4) {
          const ex = cell[k] - x, ey = cell[k + 1] - y;
          const d2 = ex * ex + ey * ey;
          if (d2 > limit) continue;
          const kind = cell[k + 2];
          // A removed point sits exactly on the kept one it came from and is
          // drawn on top of it, so on a tie the removal reason is what the
          // eye is pointing at.
          const closer = !best || d2 < best.d2 - tie;
          const onTop = best && kind > best.kind && d2 <= best.d2 + tie;
          if (closer || onTop) best = { d2, kind, i: cell[k + 3] };
        }
      }
    }
    return best;
  }

  function updateTooltip() {
    hoverFrame = 0;
    if (!map || !tooltip || !hoverPoint || moving) { hideTooltip(); return; }
    const hit = nearest(hoverPoint.x, hoverPoint.y);
    if (!hit) { hideTooltip(); return; }

    let value, unit;
    if (hit.kind === 1) {
      value = "Removed";
      unit = overlay?.groups?.[hit.i] || "";
    } else {
      const raw = data?.values?.[hit.i];
      value = raw == null ? "no value" : Units.num((data._convert || ((v) => v))(raw));
      unit = raw == null ? "" : unitLabel;
    }
    tooltip.querySelector(".v").textContent = value;
    tooltip.querySelector(".u").textContent = unit;
    tooltip.hidden = false;

    // Keep the box inside the pane: near the right or bottom edge it flips
    // to the other side of the cursor instead of being clipped.
    const size = map.getSize();
    const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    let left = hoverPoint.x + 14, top = hoverPoint.y + 14;
    if (left + w > size.x - 4) left = hoverPoint.x - w - 10;
    if (top + h > size.y - 4) top = hoverPoint.y - h - 10;
    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${Math.max(0, top)}px`;
  }

  /* ---------------------------------------------------------- polygons --- */

  /* Polygons: boundary, trial strips, prescription grid. */
  function setPolygons(rings, options = {}) {
    if (!overlayGroup) return;
    overlayGroup.clearLayers();
    if (!rings?.length) return;
    const style = {
      color: options.color || getComputedStyle(document.documentElement)
        .getPropertyValue("--accent").trim(),
      weight: options.weight ?? 1.5,
      fill: options.fill ?? false,
      fillOpacity: options.fillOpacity ?? 0.15,
    };
    for (const ring of rings) {
      const latlngs = ring.map(([lon, lat]) => [lat, lon]);
      L.polygon(latlngs, style).addTo(overlayGroup);
    }
  }

  /* Trial strips, coloured by rate, labelled on click. */
  function setFeatures(collection, rateProperty = "rate") {
    if (!overlayGroup) return;
    overlayGroup.clearLayers();
    if (!collection?.features?.length) return;

    const rates = collection.features
      .map((f) => f.properties?.[rateProperty])
      .filter((v) => v != null && isFinite(v));
    const min = Math.min(...rates), max = Math.max(...rates);
    const span = (max - min) || 1;

    L.geoJSON(collection, {
      style: (feature) => {
        const value = feature.properties?.[rateProperty];
        const c = rampColor(value == null ? 0.5 : (value - min) / span);
        return {
          color: `rgb(${c.join(",")})`, weight: 1.5,
          fillColor: `rgb(${c.join(",")})`, fillOpacity: 0.55,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties || {};
        const lines = Object.entries(p)
          .map(([k, v]) => `<b>${k}</b>: ${typeof v === "number" ? Units.num(v) : v}`);
        layer.bindPopup(lines.join("<br>"), { className: "strip-popup" });
      },
    }).addTo(overlayGroup);
    return { min, max };
  }

  /* An AB line over whatever is already drawn, with A and B markers: the visual
   * check that the line's direction matches the strips'. */
  function addLine(points, label) {
    if (!overlayGroup || !points?.length) return;
    const latlngs = points.map(([lon, lat]) => [lat, lon]);
    const accent = getComputedStyle(document.documentElement)
      .getPropertyValue("--text").trim();

    L.polyline(latlngs, { color: accent, weight: 3, opacity: 0.9, dashArray: "10 6" })
      .bindTooltip(label || "AB line", { permanent: false })
      .addTo(overlayGroup);

    for (const [index, name] of [[0, "A"], [latlngs.length - 1, "B"]]) {
      L.marker(latlngs[index], {
        icon: L.divIcon({
          className: "",
          html: `<div style="background:${accent};color:var(--surface);
                 border-radius:50%;width:20px;height:20px;display:flex;
                 align-items:center;justify-content:center;font:600 11px system-ui;
                 box-shadow:0 1px 4px rgba(0,0,0,.4)">${name}</div>`,
          iconSize: [20, 20], iconAnchor: [10, 10],
        }),
      }).addTo(overlayGroup);
    }
  }

  function clearOverlays() { overlayGroup?.clearLayers(); }

  /* ------------------------------------------- raster and named layers --- */

  /* An image the app itself serves — the terrain analyser draws each of its
   * layers as a web-mercator PNG — placed by the corners the API gives.
   *
   * It goes in a pane of its own, below the one Leaflet draws vectors in,
   * because panes stack as wholes: in the shared overlay pane the image
   * would cover the contour lines and the feature outlines that are there
   * to be read against it. The same overlay is re-pointed rather than
   * replaced, so switching from slope to wetness does not flash the
   * imagery through between two layers of one analysis — the box is the
   * same for all of them. */
  function setImage(url, bounds, options = {}) {
    if (!map || !url || !bounds) return null;
    if (!map.getPane(IMAGE_PANE)) map.createPane(IMAGE_PANE).style.zIndex = 350;
    const opacity = options.opacity ?? 0.75;
    // An image the server refuses is a blank map and no other sign, so the
    // caller is told rather than left to wonder why the field went away.
    imageFailed = options.onError || null;
    if (imageLayer) {
      imageLayer.setBounds(L.latLngBounds(bounds));
      imageLayer.setOpacity(opacity);
      // Leaflet re-requests on every setUrl and these images are served
      // no-store, so pointing at the same URL again — which every redraw of
      // the panel beside the map does — would fetch the same PNG each time.
      if (url !== imageUrl) imageLayer.setUrl(url);
    } else {
      imageLayer = L.imageOverlay(url, bounds, {
        opacity, pane: IMAGE_PANE, interactive: false,
      }).addTo(map);
      imageLayer.on("error", () => imageFailed?.());
    }
    imageUrl = url;
    return imageLayer;
  }

  function setImageOpacity(opacity) { imageLayer?.setOpacity(opacity); }

  function clearImage() {
    imageFailed = null;
    imageUrl = "";
    if (!imageLayer) return;
    map?.removeLayer(imageLayer);
    imageLayer = null;
  }

  /* Vector layers that come and go one at a time, each under its own name:
   * the contours, the hills and hollows, the drainage, the profile line.
   * `setPolygons` and `setFeatures` share a single group that is cleared as
   * a whole, which is right for one drawing and wrong here — turning the
   * contours off would take the hills away with them.
   *
   * The options are plain data so that the caller never touches Leaflet:
   * `style(properties)` returns the stroke and fill, `popup(properties)` the
   * HTML of the popup, and `tooltip(properties)` either a string or
   * `{ text, permanent }` — permanent is how a contour labels itself. */
  function setGeoJson(name, collection, options = {}) {
    if (!map) return null;
    clearGeoJson(name);
    if (!collection?.features?.length) return null;
    const styleFor = (feature) => options.style?.(feature?.properties || {}) || {};
    const layer = L.geoJSON(collection, {
      interactive: !clickThrough,
      style: styleFor,
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, styleFor(feature)),
      onEachFeature: (feature, target) => {
        const properties = feature.properties || {};
        const popup = options.popup?.(properties);
        if (popup) target.bindPopup(popup, { className: "strip-popup" });
        const tip = options.tooltip?.(properties);
        const { text, ...rest } = typeof tip === "string" ? { text: tip } : (tip || {});
        if (text) target.bindTooltip(String(text), { direction: "center", ...rest });
      },
    }).addTo(map);
    named.set(name, layer);
    return layer;
  }

  function clearGeoJson(name) {
    const layer = named.get(name);
    if (!layer) return;
    map?.removeLayer(layer);
    named.delete(name);
  }

  function clearNamed() {
    for (const name of [...named.keys()]) clearGeoJson(name);
  }

  /* A tool that collects points from the map — the elevation profile — needs
   * every click, and a click inside a hill's outline is taken by the hill:
   * Leaflet answers it with the popup and the map never hears it. Turning
   * the layers non-interactive for the length of the tool keeps them visible
   * and out of the way, which is better than hiding what the line is being
   * drawn across. */
  function setClickThrough(on) {
    clickThrough = !!on;
    if (clickThrough) map?.closePopup();
    for (const layer of named.values()) {
      layer.eachLayer((child) => { child.options.interactive = !clickThrough; });
    }
  }

  /* Centre on a feature the user picked from a table. The zoom only comes in
   * when the view is further out than the one asked for: a feature is read
   * against the ground around it, and a hill zoomed to fill the screen has
   * lost the field it stands above. */
  function panTo(lon, lat, zoom) {
    if (!map || !Number.isFinite(lon) || !Number.isFinite(lat)) return;
    map.setView([lat, lon], Math.max(map.getZoom(), zoom ?? 0), { animate: true });
  }

  function fit(bounds) {
    if (!map || !bounds || bounds.length !== 4) return;
    const [west, south, east, north] = bounds;
    if (![west, south, east, north].every(Number.isFinite)) return;
    map.fitBounds([[south, west], [north, east]], { padding: [30, 30], maxZoom: 19 });
  }

  function fitOverlays() {
    if (!overlayGroup) return;
    const layers = overlayGroup.getLayers();
    if (!layers.length) return;
    const group = L.featureGroup(layers);
    map.fitBounds(group.getBounds(), { padding: [30, 30], maxZoom: 19 });
  }

  function onClick(handler) { map.on("click", handler); }
  function offClick(handler) { map.off("click", handler); }
  function instance() { return map; }

  return {
    init, destroy, invalidateSize, setPoints, clearPoints, setOverlay, clearOverlay,
    setPolygons, setFeatures, addLine, clearOverlays,
    setImage, setImageOpacity, clearImage, setGeoJson, clearGeoJson, clearNamed,
    setClickThrough, panTo,
    fit, fitOverlays, setBasemap, rampCss, rampColor, draw, onClick, offClick, instance,
  };
}

/* Lock two views together: pan or zoom either one and the other follows.
 * The shared flag stops the echo — A moves B, B fires 'move', B would move A,
 * which would fire again. Returns the function that unlinks them. */
function linkMapViews(a, b) {
  let syncing = false;
  const follow = (source, target) => () => {
    const from = source.instance(), to = target.instance();
    if (syncing || !from || !to) return;
    syncing = true;
    // `reset` sets the exact centre; a plain pan truncates to whole pixels
    // and the two maps drift apart by a pixel on every move.
    try { to.setView(from.getCenter(), from.getZoom(), { animate: false, reset: true }); }
    finally { syncing = false; }
  };
  const ab = follow(a, b), ba = follow(b, a);
  a.instance().on("move", ab);
  b.instance().on("move", ba);
  return () => {
    a.instance()?.off("move", ab);
    b.instance()?.off("move", ba);
  };
}

/* The main map. App.init calls MapView.init("map", "point-canvas"), which
 * these defaults also cover. */
const MapView = createMapView("map", "point-canvas");
