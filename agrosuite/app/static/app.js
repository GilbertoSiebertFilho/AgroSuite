/* AgroSuite — user interface.
 *
 * Two tracks share one set of tabs. Plan a trial: a field comes in, the
 * strips and the AB line go out to the monitor. Evaluate a trial: the
 * as-applied and the yield map come in, the optimum rate and what it is
 * worth come out. Neither is an order that has to be followed — every tab is
 * clickable at any time, a tab that cannot do its work yet says what is
 * missing and offers the one button that fixes it, and every tab ends with
 * what it can produce on its own, because stopping there is a legitimate way
 * to finish. State lives in a single object; each tab redraws the right
 * panel from it. */

const App = {
  state: {
    catalog: null,
    units: null,
    datasets: [],
    selectedId: null,
    selected: null,
    colorColumn: "value",
    tab: "dados",
    design: null,
    reports: {},
    drawing: null,
    projectFile: null,   // the .agrosuite file open or last saved, as the server reports it
    projectOpen: false,  // the path dialog is picking a project, not data
    folderPick: false,   // ...or the folder projects are kept in
    saving: false,       // a project save is on its way
    recent: null,
    settings: null,      // the projects folder and the auto-save switch, from the server
    autosaveProblem: null, // the failure already reported, so it is reported once
    autosaveSeen: null,  // the auto-saved file the panel has already been told about
    view: null,          // where the reader is, as the server last heard it
    viewTimer: null,
    machinePicked: {},   // the machine picked on each tab, by name: a redraw must not forget it
    machinePlaced: {},   // what the app put in a physical field, with its exact metric number
    terrain: null,       // the relief analysis on screen: {id, summary, layers, ...}
    terrainRefusal: {},  // dataset id -> why its relief cannot be read, in the analyser's words
    terrainProbing: null,
    terrainDraw: null,   // a profile line being clicked out on the map
    terrainOnMap: false, // the relief owns the map: the points would cover it
  },

  /* ---------------------------------------------------------------- API */

  async api(path, options = {}) {
    const response = await fetch(path, {
      headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
      ...options,
      body: options.body instanceof FormData ? options.body
        : options.body ? JSON.stringify(options.body) : undefined,
    });
    let payload = null;
    try { payload = await response.json(); } catch { /* no body */ }
    if (!response.ok) {
      // A 422 from the request model is a list of {loc, msg}; read out as
      // "passes_per_strip: Input should be a valid integer" it can be acted
      // on, where the list itself would show as "[object Object]".
      const detail = Array.isArray(payload?.detail)
        ? payload.detail.map((e) => `${(e.loc || []).filter((part) => part !== "body")
            .join(".")}: ${e.msg}`).join(" ")
        : payload?.detail;
      const error = new Error(detail || `${response.status} ${response.statusText}`);
      // A 409 is a question ("replace it?"), not a failure; callers need the
      // status to tell the two apart.
      error.status = response.status;
      throw error;
    }
    return payload;
  },

  /* ------------------------------------------------------------- toasts */

  /* ``action`` is {label, run}: something the toast offers to do about what
   * it is reporting — undoing an auto-resume, say. An offer has to outlast
   * a glance, so a toast carrying one stays up four times as long. */
  toast(title, message = "", kind = "", action = null) {
    const box = document.getElementById("toasts");
    const node = document.createElement("div");
    node.className = `toast ${kind}`;
    node.innerHTML = `<div class="t"></div><div class="m"></div>`;
    node.querySelector(".t").textContent = title;
    node.querySelector(".m").textContent = message;
    if (action) {
      const button = document.createElement("button");
      button.className = "small act";
      button.textContent = action.label;
      button.addEventListener("click", () => { node.remove(); action.run(); });
      node.appendChild(button);
    }
    box.appendChild(node);
    setTimeout(() => node.remove(), action ? 20000 : kind === "error" ? 11000 : 5000);
  },

  async busy(node, fn) {
    node?.classList.add("busy");
    try { return await fn(); }
    catch (err) { this.toast("That did not work", err.message, "error"); return null; }
    finally { node?.classList.remove("busy"); }
  },

  /* ---------------------------------------------------------------- init */

  async init() {
    MapView.init("map", "point-canvas");

    try {
      this.state.catalog = await this.api("/api/catalog");
      this.state.units = this.state.catalog.units;
      Units.init(this.state.units, this.state.units.default_preset);
      // A reload starts the pickers at the default preset while the server
      // may still be holding the set from before it; saying so once puts the
      // sentences and the tables back in step.
      await this.sendDisplayUnits();
    } catch (err) {
      this.toast("Server unavailable", err.message, "error");
      return;
    }

    this.buildUnitPreset();
    this.bindTopbar();
    this.bindImport();
    this.bindDialogs();
    this.bindMachineDialog();
    this.bindProjectFiles();
    this.bindAutosave();
    this.renderTab();
    this.refreshDatasets().then(async () => {
      await this.refreshProject();
      // The recent list first, because it brings the view the server holds:
      // a reload keeps the session but not the page's selection, and so does
      // a restart, after which the session is the one the app reopened by
      // itself. Either way the page comes back where the work was left.
      await this.refreshRecent();
      await this.restoreView();
    });
    this.refreshSettings();
    this.watchAutosave();
  },

  bindTopbar() {
    document.getElementById("steps").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (button) this.openTab(button.dataset.tab);
    });

    document.getElementById("theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const current = root.getAttribute("data-theme");
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.setAttribute("data-theme", current ? (current === "dark" ? "light" : "dark")
        : (dark ? "light" : "dark"));
      MapView.draw();
      this.state.compare?.view.draw();
      this.renderTab();
    });

    document.getElementById("unit-details").addEventListener("click", () => this.openUnitsDialog());

    document.getElementById("color-column").addEventListener("change", (event) => {
      this.state.colorColumn = event.target.value;
      this.loadMap();
    });
    document.getElementById("show-basemap").addEventListener("change", (event) => {
      MapView.setBasemap(event.target.checked);
      this.state.compare?.view.setBasemap(event.target.checked);
    });
    document.getElementById("btn-fit").addEventListener("click", () => {
      if (this.state.selected?.bounds) MapView.fit(this.state.selected.bounds);
      else MapView.fitOverlays();
    });
  },

  buildUnitPreset() {
    const select = document.getElementById("unit-preset");
    select.innerHTML = "";
    for (const [key, preset] of Object.entries(this.state.units.presets)) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = preset.label;
      option.title = preset.description;
      select.appendChild(option);
    }
    select.value = Units.get().preset;
    select.addEventListener("change", (event) => {
      Units.applyPreset(event.target.value);
      this.toast("Units", this.state.units.presets[event.target.value].description);
      this.loadMap();
      this.renderTab();
      this.refreshPhrasing();
    });
  },

  /* ----------------------------------------------------------- datasets */

  async refreshDatasets() {
    const payload = await this.api("/api/datasets").catch(() => null);
    if (!payload) return;
    this.state.datasets = payload.datasets;
    this.forgetGoneReports();
    // The selected dataset's detail is fetched once, when it is picked; a
    // cleaning or an analysis run afterwards flips the flags the tabs read
    // to decide whether there is a stored report to fetch. A tab redrawn
    // after its cached result was pruned would otherwise believe there is
    // nothing to fetch. The list carries the same summary, fresh.
    const current = payload.datasets.find((d) => d.id === this.state.selectedId);
    if (current && this.state.selected) Object.assign(this.state.selected, current);
    this.renderDatasetList();
    this.refreshProject();
  },

  /* Nothing selected: the map, the overlays and the legend that belonged to
   * the selection go with it. */
  clearSelection() {
    this.state.selectedId = null;
    this.state.selected = null;
    MapView.clearPoints();
    MapView.clearOverlays();
    this.clearRemovedOverlay();
    document.getElementById("legend").hidden = true;
  },

  /* A cached cleaning result names the clean copy and the removed records it
   * produced; once either is deleted those ids point at nothing, and the
   * print and "view" buttons drawn from the cache would ask the server for a
   * dataset it no longer has. Dropping the entry makes the tab fetch the
   * stored report again and rebuild the shape from the datasets that are
   * still there — the original keeps its copy of the report, so printing
   * falls back to it, as it does after a reload. */
  forgetGoneReports() {
    const alive = new Set(this.state.datasets.map((d) => d.id));
    for (const [key, cached] of Object.entries(this.state.reports)) {
      const ids = [key.split(":")[0], cached?.clean?.id, cached?.removed?.id].filter(Boolean);
      if (ids.some((id) => !alive.has(id))) delete this.state.reports[key];
    }
    // The relief analysis and the refusals belong to their datasets too: a
    // file removed from the session takes its relief off the map with it.
    if (this.state.terrain && !alive.has(this.state.terrain.id)) {
      this.state.terrain = null;
      this.leaveTerrainMap();
    }
    for (const id of Object.keys(this.state.terrainRefusal)) {
      if (!alive.has(id)) delete this.state.terrainRefusal[id];
    }
  },

  renderDatasetList() {
    const box = document.getElementById("dataset-list");
    if (!this.state.datasets.length) {
      box.innerHTML = '<div class="empty">Nothing loaded yet.</div>';
      return;
    }
    box.innerHTML = "";
    for (const item of this.state.datasets) {
      const card = document.createElement("div");
      card.className = "card";
      card.setAttribute("aria-selected", String(item.id === this.state.selectedId));
      const origin = { clean: "clean", clean_removed: "removed", units: "converted",
                       demo: "demo", upload: "uploaded", path: "disk" }[item.origin] || item.origin;
      card.innerHTML = `
        <div class="title">${this.escape(item.label)}</div>
        <div class="meta">
          <span class="tag ${item.origin === "clean" ? "accent" : ""}">${origin}</span>
          <span>${Units.num(item.rows, 0)} rec.</span>
          <span>${this.escape(item.meta.brand_label)}</span>
        </div>`;
      card.addEventListener("click", () => this.selectDataset(item.id));
      box.appendChild(card);
    }
  },

  async selectDataset(id) {
    // The comparison and the removed-records overlay belong to one cleaning;
    // a new selection means a new map. Left in place, the overlay's dots would
    // sit on the next dataset and the tooltip would call its records removed.
    if (this.state.compare) this.exitCompare({ reload: false });
    this.clearRemovedOverlay();
    this.state.selectedId = id;
    const detail = await this.api(`/api/datasets/${id}`).catch((err) => {
      this.toast("That did not work", err.message, "error");
      return null;
    });
    if (!detail) return;
    this.state.selected = detail;

    const columns = detail.numeric_columns?.length ? detail.numeric_columns : ["value"];
    this.state.colorColumn = columns.includes(this.state.colorColumn)
      ? this.state.colorColumn : columns[0];

    this.renderDatasetList();
    this.buildColorColumns(columns);
    await this.loadMap();
    if (detail.bounds) MapView.fit(detail.bounds);
    this.renderTab();
    this.noteView();
  },

  buildColorColumns(columns) {
    const select = document.getElementById("color-column");
    select.innerHTML = "";
    for (const column of columns) {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = this.state.catalog.columns[column] || column;
      select.appendChild(option);
    }
    select.value = this.state.colorColumn;
  },

  async loadMap() {
    if (!this.state.selectedId) return;
    // In compare mode the two panes are painted together, on one scale.
    if (this.state.compare) return this.paintCompare();
    // The Terrain tab has the map: the point canvas is drawn over it, so
    // painting the points now would hide the relief the tab is about.
    if (this.state.terrainOnMap) return;
    const column = this.state.colorColumn;
    const payload = await this.api(
      `/api/datasets/${this.state.selectedId}/map?column=${encodeURIComponent(column)}`
    ).catch(() => null);
    if (!payload) return;

    const { conv, unit } = Units.forColumn(column, this.state.selected?.meta?.operation);
    const scale = MapView.setPoints(payload, conv, { unit });
    if (payload.polygons?.length) MapView.setPolygons(payload.polygons);
    else if (this.state.tab !== "ensaio") MapView.clearOverlays();

    document.getElementById("map-status").textContent = this.pointsStatus(payload);
    this.renderLegend(column, unit, scale);
  },

  /* Past MAP_POINT_LIMIT the server sends a systematic sample, and `count`
   * is the size of the sample, not of the dataset. Any count shown to the
   * user has to say so, or two datasets on either side of the limit read
   * as if the smaller one had more points. */
  pointsStatus(payload) {
    return payload.sampled
      ? `${Units.num(payload.count, 0)} of ${Units.num(payload.total, 0)} points (sampled)`
      : `${Units.num(payload.count, 0)} points`;
  },

  /* The legend reads in the displayed unit, like the colours: `scale` is the
   * one setPoints returned, already converted. */
  renderLegend(column, unit, scale) {
    const legend = document.getElementById("legend");
    if (scale) {
      legend.hidden = false;
      document.getElementById("legend-title").textContent =
        `${this.state.catalog.columns[column] || column}${unit ? ` (${unit})` : ""}`;
      document.getElementById("legend-ramp").style.background = MapView.rampCss();
      document.getElementById("legend-min").textContent = Units.num(scale.low);
      document.getElementById("legend-max").textContent = Units.num(scale.high);
    } else {
      legend.hidden = true;
    }
  },

  /* ------------------------------------------------------------ helpers */

  /* Safe as text and inside a quoted attribute alike. textContent alone
   * leaves quotes as they are, and a name such as 'S780 "40 ft"' put in a
   * value="…" would end at its first quote. */
  escape(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  },

  field(label, inputHtml, hint) {
    return `<label class="field"><span>${label}</span>${inputHtml}` +
      (hint ? `<span class="hint tight" style="margin-top:3px;display:block">${hint}</span>` : "") +
      `</label>`;
  },

  numberInput(id, value, step = "any", min = null) {
    return `<input type="number" id="${id}" value="${value ?? ""}" step="${step}"` +
      (min != null ? ` min="${min}"` : "") + `>`;
  },

  selectInput(id, options, selected) {
    const items = options.map(([value, label]) =>
      `<option value="${this.escape(value)}"${value === selected ? " selected" : ""}>${
        this.escape(label)}</option>`).join("");
    return `<select id="${id}">${items}</select>`;
  },

  /* Round to the nearest "round number" in the same order of magnitude:
   * 44.8 becomes 50; 4.48 becomes 5; 448 becomes 450. */
  roundStep(value) {
    if (!isFinite(value) || value <= 0) return 1;
    const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
    const normalized = value / magnitude;
    const nice = normalized <= 1.5 ? 1 : normalized <= 3.5 ? 2.5 : normalized <= 7.5 ? 5 : 10;
    return Math.round(nice * magnitude * 100) / 100;
  },

  value(id) { return document.getElementById(id)?.value; },
  number(id, fallback = null) {
    const raw = document.getElementById(id)?.value;
    if (raw === "" || raw == null) return fallback;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : fallback;
  },
  checked(id) { return document.getElementById(id)?.checked ?? false; },

  columnOptions(extra = []) {
    const columns = this.state.selected?.numeric_columns || [];
    const labels = this.state.catalog.columns;
    return [...extra, ...columns.map((c) => [c, labels[c] || c])];
  },

  /* ------------------------------------------------------- tab navigation */

  goToTab(tab) {
    this.state.tab = tab;
    for (const b of document.querySelectorAll("#steps button")) {
      b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    }
    this.renderTab();
  },

  /* --------------------------------------- what a tab needs, what it gives */

  /* A tab that cannot do its work yet says so in its own panel and offers the
   * one button that fixes it. Greying the tab out, or drawing an empty panel,
   * would leave the user to guess which of the missing pieces is the one this
   * tab is waiting for. */
  missingPanel(title, what, action) {
    return `
      <div class="panel">
        <h3>${this.escape(title)}</h3>
        <div class="note">${this.escape(what)}</div>
        <button class="primary wide" style="margin-top:10px" data-cta="${action.cta}">
          ${this.escape(action.label)}</button>
        ${action.hint ? `<p class="hint tight">${this.escape(action.hint)}</p>` : ""}
      </div>`;
  },

  /* The end of every tab: what it produces on its own and how to take it
   * away. The buttons are never disabled — one pressed too early answers
   * with the step that would fill it, which is more use than a dead control. */
  producesPanel(rows, note = "") {
    return `
      <div class="panel produces">
        <h3>If you stop here</h3>
        ${note ? `<p class="hint tight">${this.escape(note)}</p>` : ""}
        ${rows.map((row) => `
          <button class="wide" style="margin-bottom:4px" data-cta="${row.cta}">
            ${this.escape(row.label)}</button>
          <p class="hint tight">${this.escape(row.hint)}</p>`).join("")}
      </div>`;
  },

  /* One place where every call to action in a panel is wired, so a panel only
   * has to name the step it wants rather than repeat the handler. */
  bindPanelActions(panel) {
    for (const button of panel.querySelectorAll("[data-cta]")) {
      button.addEventListener("click", (event) =>
        this.callToAction(button.dataset.cta, event.currentTarget));
    }
  },

  async callToAction(action, button) {
    const [name, argument] = String(action).split(":");
    const selected = this.state.selectedId;

    if (name === "load") { document.getElementById("btn-open-file").click(); return; }
    if (name === "pick") {
      const id = argument || this.state.datasets[0]?.id;
      if (!id) { this.toast("Nothing to pick", "Load a file first.", "warn"); return; }
      await this.selectDataset(id);
      return;
    }
    // Like 'pick', but for a layer the app has just made: the Data tab is
    // where a new dataset describes itself, and leaving the user on the tab
    // that produced it would answer "show me" with the form that made it.
    if (name === "inspect") {
      const id = argument || selected;
      if (!id) { this.toast("Nothing to show", "Load a file first.", "warn"); return; }
      // The tab moves first: a dataset picked while a tab that cannot use it
      // is on screen would have that tab draw its refusal on the way past.
      this.goToTab("dados");
      await this.selectDataset(id);
      return;
    }
    if (name === "draw") { this.goToTab("ensaio"); this.startDrawing(); return; }
    if (name === "demo-terrain") { document.getElementById("btn-demo-terrain").click(); return; }
    // The relief's two products are made on the Terrain tab itself, so these
    // lead to the control rather than to another tab; pressed before there is
    // an analysis, they say what would fill them.
    if (name === "terrain-zones" || name === "terrain-export") {
      this.goToTab("terrain");
      const id = name === "terrain-zones" ? "btn-terrain-zones" : "btn-terrain-export";
      const control = document.getElementById(id);
      if (!control) {
        this.toast("Nothing to make it from",
          "Run \"Analyse the relief\" on this tab first.", "warn");
        return;
      }
      control.scrollIntoView({ block: "center", behavior: "smooth" });
      control.focus();
      return;
    }
    if (name === "prices") {
      const field = document.getElementById("difm-price") || document.getElementById("price-value");
      if (!field) { this.goToTab("difm"); return; }
      field.scrollIntoView({ block: "center", behavior: "smooth" });
      field.focus();
      return;
    }
    if (name === "print") {
      const id = argument || selected;
      if (!id) { this.toast("Nothing to print", "Pick a file first.", "warn"); return; }
      await this.printReport(id, button);
      return;
    }
    if (name === "export-file") {
      const id = argument || selected;
      if (!id) { this.toast("Nothing to export", "Pick a file first.", "warn"); return; }
      await this.exportDataset(id);
      return;
    }
    if (name === "print-clean") {
      const id = this.cleanCopyId();
      if (!id && !this.state.selected?.has_clean_report) {
        this.toast("No cleaning report yet",
          "Run the cleaning above and the report comes with it.", "warn");
        return;
      }
      await this.printReport(id || selected, button);
      return;
    }
    if (name === "export-clean") {
      const id = this.cleanCopyId();
      if (!id) {
        this.toast("No clean copy yet",
          "Run the cleaning and it produces one, beside the original.", "warn");
        return;
      }
      await this.exportDataset(id);
      return;
    }
    if (name === "export-package") {
      this.state.exportMode = "package";
      this.goToTab("exportar");
      return;
    }
    if (name === "prescription") {
      const report = this.state.reports[`${selected}:difm`];
      if (!report) {
        this.toast("Nothing to build from",
          "The prescription comes out of the analysis; run it first.", "warn");
        return;
      }
      this.prescriptionFromDifm(report);
    }
  },

  /* Hand a dataset to the Export tab ready to write: individual files, this
   * dataset as the source. "The file as it stands" is a product in its own
   * right — the data read, normalized and in the units the user declared. */
  async exportDataset(id) {
    if (id && id !== this.state.selectedId) await this.selectDataset(id);
    this.state.exportMode = "files";
    this.state.exportSource = "dataset";
    this.goToTab("exportar");
  },

  /* The clean copy of what is selected: itself, if a clean copy is what is
   * selected, and otherwise the last one the cleaning produced from it. */
  cleanCopyId() {
    const d = this.state.selected;
    if (!d) return null;
    if (d.origin === "clean") return d.id;
    return this.cleanResultFor(d.id, null).clean?.id || null;
  },

  /* ------------------------------------------------ right panel rendering */

  renderTab() {
    const panel = document.getElementById("right-panel");
    // The comparison only makes sense next to its cleaning report.
    if (this.state.compare && this.state.tab !== "limpeza") this.exitCompare();
    // The relief belongs to the Terrain tab and to the file it was read
    // from: anywhere else the map is the points again, and another file's
    // points over this file's relief would be two fields in one picture.
    if (this.state.terrainOnMap && (this.state.tab !== "terrain" || !this.terrainAnalysis())) {
      this.leaveTerrainMap();
    }
    const renderers = {
      dados: () => this.tabDados(panel),
      terrain: () => this.tabTerrain(panel),
      limpeza: () => this.tabLimpeza(panel),
      difm: () => this.tabDifm(panel),
      ensaio: () => this.tabEnsaio(panel),
      exportar: () => this.tabExportar(panel),
    };
    (renderers[this.state.tab] || renderers.dados)();
    // Every panel's calls to action are wired in one place, after the tab
    // has drawn itself, so a tab only names the step it is asking for.
    this.bindPanelActions(panel);
  },
};

/* ======================================================================
 * Data
 * ==================================================================== */

Object.assign(App, {
  tabDados(panel) {
    const d = this.state.selected;
    if (!d) {
      const loaded = this.state.datasets.length;
      panel.innerHTML = loaded
        ? this.missingPanel(
            "Data",
            `${loaded} file(s) are open, but none is picked. The panel shows one at a `
            + "time: what came in, whether the units are what they seem, and how the "
            + "data looks.",
            { cta: "pick", label: "Show the first file" })
        : this.missingPanel(
            "Data",
            "Nothing is open yet. This tab reads what came off the monitor — "
            + "shapefile, CSV, ISOXML, a card or a zip — and gives it a first look: "
            + "coverage, timeline, columns, data quality and the units.",
            { cta: "load", label: "Load a file",
              hint: "Or try it on made-up data: the Harvest demo and the Trial demo "
                    + "buttons are on the left." });
      return;
    }

    const meta = d.meta;
    const stats = d.stats || {};
    const areaUnit = Units.label.area();
    const confidence = meta.extra?.brand_confidence;

    panel.innerHTML = `
      ${this.preflightPanel(d.reports_data?.preflight)}
      <div class="panel">
        <h3>Source</h3>
        <div class="stat-grid">
          <div class="stat"><div class="k">Monitor identified</div>
            <div class="v" style="font-size:13px">${this.escape(meta.brand_label)}</div>
            <div class="d">${confidence != null
              ? `${Math.round(confidence * 100)}% confidence` : "declared"}</div></div>
          <div class="stat"><div class="k">Operation</div>
            <div class="v" style="font-size:13px">${this.escape(meta.operation_label)}</div>
            <div class="d">${this.escape(meta.source_format)}</div></div>
          <div class="stat"><div class="k">Records</div>
            <div class="v">${Units.num(d.rows, 0)}</div>
            <div class="d">${d.metric_crs || ""}</div></div>
          <div class="stat"><div class="k">Area worked</div>
            <div class="v">${Units.num(Units.convert.area(d.area_ha))}</div>
            <div class="d">${areaUnit}</div></div>
        </div>
        <div class="row tight" style="margin-top:10px">
          <button class="small" id="btn-declare-units">File units</button>
          ${meta.brand === "augmenta"
            ? '<button class="small" id="btn-augmenta">Vigour vs rate</button>' : ""}
        </div>
      </div>

      <div class="panel">
        <h3>Main variable · ${this.escape(
          this.state.catalog.columns[this.state.colorColumn] || this.state.colorColumn)}</h3>
        ${this.statsBlock(stats, this.state.colorColumn)}
      </div>

      ${meta.extra?.jd_card ? this.cardPanel(meta.extra.jd_card, meta.extra.jd_card_layers) : ""}

      ${meta.notes?.length ? `<div class="panel"><h3>Notes from reading the file</h3>
        ${meta.notes.map((n) => `<div class="note">${this.escape(n)}</div>`).join("")}</div>` : ""}

      ${Object.keys(meta.extra?.column_mapping || {}).length ? `
      <div class="panel"><h3>Columns recognized</h3>
        <details class="fold"><summary>${
          Object.keys(meta.extra.column_mapping).length} columns mapped</summary>
          <div class="inner scroll-x"><table class="data">
            <tr><th>In the file</th><th>Read as</th></tr>
            ${Object.entries(meta.extra.column_mapping).map(([raw, canonical]) =>
              `<tr><td>${this.escape(raw)}</td><td>${this.escape(
                this.state.catalog.columns[canonical] || canonical)}</td></tr>`).join("")}
          </table></div>
        </details>
      </div>` : ""}

      <div class="panel"><h3>First rows</h3>
        <div class="scroll-x" id="preview"></div>
      </div>

      ${this.producesPanel([
        { label: "Print the report (PDF)", cta: "print",
          hint: "The first look as a page: what the file is, what it covers and "
                + "what was found in it, in the units on screen." },
        { label: "Export this file as it stands", cta: "export-file",
          hint: "The data as read — normalized columns, declared units — written "
                + "as shapefile, CSV or GeoJSON." },
      ], "Reading a file and stopping there is a complete use of the app.")}

      <div class="panel">
        <button class="wide" id="btn-remove-dataset">Remove from session</button>
      </div>`;

    this.renderPreview(document.getElementById("preview"), d.preview);

    document.getElementById("btn-apply-units")?.addEventListener("click", () => {
      const proposed = d.reports_data?.preflight?.proposed_units;
      if (proposed) this.applyProposedUnits(proposed);
    });
    document.getElementById("btn-preflight-next")?.addEventListener("click", () => {
      const step = d.reports_data?.preflight?.next_step?.step;
      if (step === "units") this.openImportUnitsDialog();
      else this.goToStep(step || "clean");
    });
    document.getElementById("btn-declare-units")
      ?.addEventListener("click", () => this.openImportUnitsDialog());
    document.getElementById("btn-augmenta")
      ?.addEventListener("click", () => this.runAugmenta());
    document.getElementById("btn-print-preflight")?.addEventListener("click", (event) =>
      this.printReport(this.state.selectedId, event.currentTarget));
    document.getElementById("btn-remove-dataset").addEventListener("click", async () => {
      await this.api(`/api/datasets/${this.state.selectedId}`, { method: "DELETE" });
      this.clearSelection();
      await this.refreshDatasets();
      this.renderTab();
    });
  },

  /* The preliminary pass, shown the moment a file opens rather than on
   * request: the user should not have to ask whether what they just loaded is
   * usable. Alerts come first because they block everything downstream. */
  preflightPanel(report) {
    if (!report) return "";
    const proposed = report.proposed_units;
    const noteClass = { ok: "ok", warning: "warning", alert: "alert" }[report.verdict] || "";
    const icon = { ok: "✓", warning: "!", alert: "✕" };
    const order = { alert: 0, warning: 1, ok: 2 };
    const findings = [...report.findings].sort(
      (a, b) => order[a.level] - order[b.level]);

    return `
      <div class="panel">
        ${this.printableHead("First look at this file", "btn-print-preflight")}
        <div class="note ${noteClass}">${this.escape(report.summary)}</div>
        <div style="margin-top:8px">
          ${findings.map((f) => `
            <div class="req ${f.level === "ok" ? "done" : "todo"}" style="align-items:flex-start">
              <span class="mark" style="color:${
                f.level === "ok" ? "var(--ok)"
                : f.level === "warning" ? "var(--warn)" : "var(--danger)"
              }">${icon[f.level]}</span>
              <span class="txt">
                <b>${this.escape(f.title)}</b>
                <br><span class="d">${this.escape(f.detail)}</span>
                ${f.action ? `<br><span class="d" style="color:var(--text)">${
                  this.escape(f.action)}</span>` : ""}
              </span>
            </div>`).join("")}
        </div>
        <div class="row tight" style="margin-top:10px">
          ${proposed ? `<button class="primary" id="btn-apply-units">
            Apply ${this.escape(Object.entries(proposed)
              .filter(([k]) => k !== "crop").map(([, v]) => v).join(" · "))}
          </button>` : ""}
          <button class="${report.verdict === "alert" && !proposed ? "primary" : ""}"
                  id="btn-preflight-next">
            ${this.escape(report.next_step.label)}
          </button>
        </div>
        <p class="hint tight">${this.escape(report.next_step.why)}</p>
      </div>`;
  },

  /* John Deere card inventory: what can be read and what is proprietary.
   * Knowing a file exists but is closed is different from thinking the card
   * came up empty. */
  cardPanel(card, layers) {
    const roleLabels = {
      boundary: "Boundary", guidance: "AB lines",
      prescription: "Prescription", data: "Operation data",
    };
    return `
      <div class="panel">
        <h3>John Deere card</h3>
        <div class="note">${this.escape(card.summary)}</div>
        ${layers?.length ? `<h4>Layers the app can open</h4>
          <div class="scroll-x"><table class="data">
            <tr><th>File</th><th>Role</th></tr>
            ${layers.map((l) => `<tr>
              <td style="font-family:var(--mono);font-size:11px">${this.escape(l.relative)}</td>
              <td>${this.escape(roleLabels[l.role] || l.role)}</td></tr>`).join("")}
          </table></div>` : ""}
        ${card.proprietary?.length ? `<details class="fold" style="margin-top:8px">
          <summary>${card.proprietary.length} file(s) in a proprietary format</summary>
          <div class="inner">
            ${card.proprietary.map((f) => `<p class="hint tight">
              <span style="font-family:var(--mono)">${this.escape(f.relative)}</span> — ${
                this.escape(f.kind)}</p>`).join("")}
            <p class="hint tight" style="margin-top:6px">These files only open in John
            Deere's own software or on the display itself. To bring them into AgroSuite,
            export again from SMS choosing shapefile instead of GreenStar.</p>
          </div></details>` : ""}
      </div>`;
  },

  statsBlock(stats, column) {
    if (!stats || !stats.n) return '<div class="empty">No statistics.</div>';
    const { conv, unit } = Units.forColumn(column, this.state.selected?.meta?.operation);
    const cell = (label, value, decimals) =>
      `<div class="stat"><div class="k">${label}</div>
        <div class="v">${Units.num(conv(value), decimals)}</div>
        <div class="d">${unit}</div></div>`;
    return `<div class="stat-grid">
      ${cell("Mean", stats.mean)}
      ${cell("Median", stats.median)}
      ${cell("Std deviation", stats.std)}
      <div class="stat"><div class="k">Coeff. of variation</div>
        <div class="v">${Units.num(stats.cv, 1)}</div><div class="d">%</div></div>
      ${cell("Minimum", stats.min)}
      ${cell("Maximum", stats.max)}
      ${cell("5th percentile", stats.p05)}
      ${cell("95th percentile", stats.p95)}
    </div>`;
  },

  renderPreview(node, preview) {
    if (!node || !preview?.rows?.length) {
      if (node) node.innerHTML = '<div class="empty">No rows.</div>';
      return;
    }
    const columns = preview.columns.slice(0, 10);
    const head = columns.map((c) =>
      `<th>${this.escape(this.state.catalog.columns[c] || c)}</th>`).join("");
    const body = preview.rows.slice(0, 12).map((row) =>
      `<tr>${columns.map((c) => {
        const value = row[c];
        const text = typeof value === "number" ? Units.num(value) : (value ?? "—");
        return `<td class="${typeof value === "number" ? "num" : ""}">${this.escape(text)}</td>`;
      }).join("")}</tr>`).join("");
    node.innerHTML = `<table class="data"><tr>${head}</tr>${body}</table>` +
      (preview.columns.length > 10
        ? `<p class="hint">Showing 10 of ${preview.columns.length} columns.</p>` : "");
  },

  async runAugmenta() {
    const report = await this.busy(document.getElementById("right-panel"), () =>
      this.api(`/api/datasets/${this.state.selectedId}/augmenta`, { method: "POST" }));
    if (!report) return;
    if (!report.available) { this.toast("Augmenta", report.reason, "warn"); return; }

    const rows = report.classes.map((c) =>
      `<tr><td>${this.escape(c["class"])}</td><td class="num">${Units.num(c.n, 0)}</td>
       <td class="num">${Units.num(c.mean_rate)}</td>
       <td class="num">${Units.num(c.sd)}</td></tr>`).join("");
    const panel = document.getElementById("right-panel");
    const block = document.createElement("div");
    block.className = "panel";
    block.innerHTML = `<h3>Augmenta · vigour vs rate</h3>
      <div class="note ${report.relative_spread_pct < 5 ? "warning" : "ok"}">${
        this.escape(report.reading)}</div>
      <div class="stat-grid" style="margin:8px 0">
        <div class="stat"><div class="k">Correlation</div>
          <div class="v">${Units.num(report.correlation, 2)}</div>
          <div class="d">vigour vs rate</div></div>
        <div class="stat"><div class="k">Rate spread</div>
          <div class="v">${Units.num(report.relative_spread_pct, 1)}</div><div class="d">%</div></div>
      </div>
      <div class="scroll-x"><table class="data">
        <tr><th>Vigour class</th><th>n</th><th>Mean rate</th><th>SD</th></tr>${rows}
      </table></div>`;
    panel.insertBefore(block, panel.children[1]);
  },
});

/* ======================================================================
 * Terrain
 *
 * The relief of the field, read from the altitude the monitor already
 * logged: where the water sits, which way the ground falls, which end
 * washes, where it is too steep to work across. It needs no survey and no
 * drone — only a file carrying a GPS height — which is why it sits beside
 * Data, and why both tracks pass through it: the relief decides where the
 * strips go as much as it explains a yield map.
 *
 * The analyser speaks metric. Everything drawn here is converted the way
 * the rest of the app converts, with one deliberate exception, named in
 * the panel itself: the findings are sentences with their numbers written
 * into them, and converting those would mean rewriting what they say.
 * ==================================================================== */

Object.assign(App, {
  /* The layers that come and go on the map, each under its own name so one
   * can be switched off without taking the others with it. */
  TERRAIN_LAYERS: ["terrain-contours", "terrain-features", "terrain-drainage",
                   "terrain-profile"],

  /* How each kind of relief feature is drawn. A closed depression is dashed
   * and darker than an open low, because whether the hollow closes is what
   * decides if the water sits there or runs out of it. */
  TERRAIN_FEATURE_STYLE: {
    hill: { color: "#a0522d", dashArray: null },
    low: { color: "#3182bd", dashArray: null },
    depression: { color: "#1f4e79", dashArray: "4 3" },
  },

  /* ---------------------------------------------------------- the tab */

  tabTerrain(panel) {
    const d = this.state.selected;
    if (!d) {
      const loaded = this.state.datasets.length;
      panel.innerHTML = (loaded
        ? this.missingPanel(
            "Terrain",
            "The relief is read from one file's altitude, and none is picked yet.",
            { cta: "pick", label: "Show the first file" })
        : this.missingPanel(
            "Terrain",
            "Nothing is open yet. This tab reads the GPS height every monitor "
            + "file already carries and turns it into the relief of the field: "
            + "the hills, the low ground, the closed hollows where water sits, "
            + "the slope classes and which way each part faces. A DEM GeoTIFF "
            + "works too, and is read at its own resolution.",
            { cta: "load", label: "Load a file that carries the altitude",
              hint: "A yield or as-applied export logs a height on every point. "
                    + "Nothing else is needed here: no prices, no cleaning, no trial." }))
        + this.terrainDemoPanel();
      return;
    }

    // The analyser's own sentence about this file, kept from the refusal it
    // gave: it names what to open instead, which is more use than a form
    // that cannot be submitted.
    const refusal = this.state.terrainRefusal[d.id];
    if (refusal) {
      // A zone layer this analyser cut carries the id of the file it was cut
      // from, so the panel can offer that file itself. The sentence then
      // loses its closing parenthesis — the one naming the dataset id and the
      // endpoint — because the button does what it was telling the caller to
      // do, and an id and a URL are not an instruction to a person.
      const source = this.state.datasets.find(
        (item) => item.id === d.meta?.extra?.terrain_source_id);
      panel.innerHTML = source
        ? this.missingPanel("Terrain", refusal.replace(/\s*\(dataset [^)]*\)/, ""),
            { cta: `pick:${source.id}`, label: `Show ${source.label}`,
              hint: "Zones are what the analysis produced, not a layer of the "
                    + "field: re-gridding their cell centres would hand back a "
                    + "coarser copy of the relief they were cut from." })
        : this.missingPanel("Terrain", refusal,
            { cta: "load", label: "Open a file that carries the altitude",
              hint: "A yield or as-applied export from the monitor logs a GPS height "
                    + "on every point; a DEM GeoTIFF of the field works too." })
          + this.terrainDemoPanel();
      return;
    }

    if (this.state.terrainProbing === d.id) {
      panel.innerHTML = `
        <div class="panel"><h3>Terrain</h3>
          <div class="note">Looking for a usable altitude in
            ${this.escape(d.label)}…</div>
        </div>`;
      return;
    }

    const produces = this.producesPanel([
      { label: "Print the report (PDF)", cta: "print",
        hint: "The relief on a page: the character of the field, which way it "
              + "falls, the hills, the low ground and the depressions, the slope "
              + "classes and every finding, in the units on screen." },
      { label: "Make zones for the monitor", cta: "terrain-zones",
        hint: "Landform, slope class, elevation bands or wetness as a layer the "
              + "machine can read — the relief turned into something to apply." },
      { label: "Export the layers for QGIS", cta: "terrain-export",
        hint: "GeoTIFFs on the analysis grid, the contours, the features and the "
              + "drainage, zipped with a README that names each file." },
    ], "Reading the relief and stopping there is a complete use of the app: "
       + "nothing further down the tabs is needed to act on what it found.");

    const analysis = this.terrainAnalysis();
    if (!analysis) {
      panel.innerHTML = this.terrainRunPanel(null) + produces;
      this.bindTerrainRun();
      // A file with no altitude column cannot be analysed, and asking the
      // analyser is how the panel gets the sentence that says what to open
      // instead — it refuses before any gridding, so it costs nothing.
      if (this.terrainLooksUnanalysable(d)) this.probeTerrain(d.id);
      else if (d.reports?.terrain) this.restoreTerrainReport(d);
      return;
    }

    const summary = analysis.summary;
    panel.innerHTML =
      this.terrainFindingsPanel(summary) +
      this.terrainRunPanel(analysis) +
      this.terrainFieldPanel(summary) +
      this.terrainYieldPanel(analysis) +
      this.terrainMapPanel(analysis) +
      this.terrainFeaturesPanel(summary) +
      this.terrainProfilePanel() +
      this.terrainSlopePanel(summary) +
      this.terrainAspectPanel(summary) +
      this.terrainLandformPanel(summary) +
      this.terrainZonesPanel(analysis) +
      this.terrainExportPanel() +
      produces;

    this.bindTerrainRun();
    this.bindTerrainMapControls();
    this.bindTerrainPanels();
    this.bindTerrainYield();
    this.drawTerrainCharts();
    this.showTerrainLayer();
  },

  terrainDemoPanel() {
    return this.missingPanel(
      "Or try it on made-up relief",
      "The terrain demo is a synthetic field of known relief — two hills, a "
      + "valley and a closed hollow — logged the way a monitor would log it, "
      + "GPS noise and pass-to-pass offsets included.",
      { cta: "demo-terrain", label: "Terrain demo" });
  },

  /* The analysis on screen, when it belongs to the dataset on screen: one
   * analysis is kept, and picking another file must not show that file the
   * relief of the previous one. */
  terrainAnalysis() {
    const terrain = this.state.terrain;
    return terrain && terrain.id === this.state.selectedId ? terrain : null;
  },

  /* A file with no altitude column at all, or the zone layer this analyser
   * itself produced — both are refused, and the refusal is where the
   * sentence for the panel comes from. */
  terrainLooksUnanalysable(d) {
    if (d.meta?.extra?.zones_by) return true;
    if (d.meta?.extra?.dem_path) return false;
    return !(d.columns || []).includes("elev_m");
  },

  async probeTerrain(id) {
    this.state.terrainProbing = id;
    try {
      const result = await this.api("/api/terrain/analyze", {
        method: "POST", body: { dataset_id: id },
      });
      // It could be analysed after all — an altitude under another name, a
      // raster behind the points. Keep the result rather than throwing away
      // work the user would only have to ask for again.
      this.rememberTerrain(id, result.summary);
      await this.afterTerrainRun(id);
    } catch (err) {
      if (err.status === 400) this.state.terrainRefusal[id] = err.message;
      else this.toast("That did not work", err.message, "error");
    } finally {
      if (this.state.terrainProbing === id) this.state.terrainProbing = null;
      if (this.state.tab === "terrain" && this.state.selectedId === id) this.renderTab();
    }
  },

  /* ------------------------------------------------- running the analysis */

  terrainRunPanel(analysis) {
    const lengthUnit = Units.label.length();
    const options = analysis?.summary?.options;
    const shown = (metres) => (metres == null || !isFinite(metres)
      ? "" : Number(Units.convert.length(metres).toFixed(3)));

    return `
      <div class="panel">
        <h3>${analysis ? "Analysis" : "The relief of this field"}</h3>
        ${analysis ? "" : `<p class="hint tight">${this.state.selected?.meta?.extra?.dem_path
          ? `This layer is an elevation raster: the analysis reads the file itself,
             at its own resolution, rather than the sample of it drawn on the map.`
          : `Every point in a monitor file carries a GPS height. Scattered along a
             track they say nothing; on a grid they answer where the water sits,
             which way the field falls and where the ground is too steep to work
             across.`}</p>`}
        <details class="fold">
          <summary>Options — cell, smoothing, contour interval, smallest feature</summary>
          <div class="inner">
            <div class="row tight">
              ${this.field(`Grid cell (${lengthUnit})`,
                this.numberInput("terrain-cell", shown(options?.cell_m), "any", "0"))}
              ${this.field(`Smoothing (${lengthUnit})`,
                this.numberInput("terrain-smooth", shown(options?.smooth_m), "any", "0"))}
            </div>
            <div class="row tight">
              ${this.field(`Contour interval (${lengthUnit})`,
                this.numberInput("terrain-interval", shown(options?.contour_interval_m), "any", "0"))}
              ${this.field(`Smallest feature (${lengthUnit})`,
                this.numberInput("terrain-feature", shown(options?.min_feature_height_m), "any", "0"))}
            </div>
            <p class="hint tight">An empty box is chosen from the data: the cell
            from the swath, the smoothing from the noise in the readings, the
            interval from the relief, and the smallest feature from twice the
            GPS noise — a bump smaller than the noise is the noise. Smoothing 0
            reads the raw surface.</p>
          </div>
        </details>
        <button class="primary wide" id="btn-run-terrain">
          ${analysis ? "Analyse the relief again" : "Analyse the relief"}</button>
        <p class="hint tight">The heights are gridded, the pass-to-pass offsets
        solved and removed, and everything below is read off the result.</p>
      </div>`;
  },

  bindTerrainRun() {
    document.getElementById("btn-run-terrain")
      ?.addEventListener("click", () => this.runTerrain());
    document.getElementById("btn-print-terrain")?.addEventListener("click", (event) =>
      this.printReport(this.state.selectedId, event.currentTarget));
  },

  async runTerrain() {
    const id = this.state.selectedId;
    if (!id) { this.toast("Nothing to analyse", "Pick a file first.", "warn"); return; }

    const metres = (field) => {
      const typed = this.number(field, null);
      return typed == null ? null : Units.toInternal.length(typed);
    };
    const body = { dataset_id: id };
    const asked = {
      cell_m: metres("terrain-cell"),
      smooth_m: metres("terrain-smooth"),
      contour_interval_m: metres("terrain-interval"),
      min_feature_height_m: metres("terrain-feature"),
    };
    for (const [key, value] of Object.entries(asked)) {
      if (value != null) body[key] = value;
    }
    const defaults = Object.values(asked).every((value) => value == null);

    // The whole panel greys while it runs: on a full yield map the gridding,
    // the smoothing and the hydrology take a few seconds, and every number in
    // the panel belongs to the analysis that is being replaced.
    const result = await this.busy(document.getElementById("right-panel"), async () => {
      try {
        return await this.api("/api/terrain/analyze", { method: "POST", body });
      } catch (err) {
        // With every box empty nothing the user typed can be wrong, so a
        // refusal is about the file: it becomes the panel, with the sentence
        // that names what to open instead. With a box filled in, the same
        // sentence belongs beside the box, where it can be acted on.
        if (err.status === 400 && defaults) {
          this.state.terrainRefusal[id] = err.message;
          this.state.terrain = null;
          this.renderTab();
          return null;
        }
        throw err;
      }
    });
    if (!result) return;

    this.rememberTerrain(id, result.summary);
    await this.afterTerrainRun(id);
    this.renderTab();
    const character = result.summary.character?.label || "analysed";
    this.toast("Relief analysed",
      `${character[0].toUpperCase()}${character.slice(1)}: ` +
      `${Units.num(Units.convert.length(result.summary.elevation.relief_m), 1)} ` +
      `${Units.label.length()} between the highest and the lowest ground.`);
  },

  rememberTerrain(id, summary) {
    const previous = this.state.terrain?.id === id ? this.state.terrain : null;
    delete this.state.terrainRefusal[id];
    this.state.terrain = {
      id,
      summary,
      layers: null,
      layersGone: null,
      // What the user had chosen stays chosen across a re-run: the point of
      // running again is usually to see the same layer under other options.
      layerKey: previous?.layerKey || "elevation",
      opacity: previous?.opacity ?? 0.75,
      hillshade: previous?.hillshade ?? false,
      contoursOn: previous?.contoursOn ?? false,
      featuresOn: previous?.featuresOn ?? true,
      drainageOn: previous?.drainageOn ?? false,
      contours: null,
      features: null,
      profile: null,
      // The value layer stays chosen across a re-run, like the map layer:
      // running again is usually to see the same comparison under other
      // options. The comparison itself does not — it was read off the grid
      // that has just been replaced.
      valuesId: previous?.valuesId || null,
      yieldRelief: null,
      zones: previous?.zones || null,
      exported: null,
      fitted: false,
      // The layer PNGs live under one URL per key and change whenever the
      // analysis is re-run; the stamp in the query is what stops the browser
      // from showing the previous relief under the new options.
      stamp: Date.now(),
    };
  },

  /* Everything the map needs, fetched once per analysis: the layer list with
   * its legends, and the feature outlines the popups are filled from. */
  async afterTerrainRun(id) {
    const analysis = this.state.terrain;
    if (!analysis || analysis.id !== id) return;
    try {
      analysis.layers = (await this.api(`/api/terrain/${id}/layers`)).layers;
    } catch (err) {
      // The arrays are evicted after four other analyses; the summary
      // outlives them, so the panel stays and only the map goes.
      analysis.layersGone = err.message;
    }
    analysis.features = await this.api(`/api/terrain/${id}/features`).catch(() => null);
    if (analysis.contoursOn) await this.loadTerrainContours(null);
    await this.refreshDatasets();
  },

  /* After a reload the server still holds the analysis, and the summary is
   * saved with the dataset: the panel comes back without re-running. */
  async restoreTerrainReport(d) {
    if (this.state.terrainRestoring === d.id) return;
    this.state.terrainRestoring = d.id;
    const report = await this.storedReport("terrain");
    this.state.terrainRestoring = null;
    if (!report || this.state.selectedId !== d.id || this.state.tab !== "terrain") return;
    this.rememberTerrain(d.id, report);
    // The comparison is saved beside the relief and is just as complete:
    // bringing back one without the other would ask the user to press
    // Compare again on numbers the session still holds.
    const comparison = await this.storedReport("terrain_yield");
    if (comparison && this.state.terrain?.id === d.id) {
      this.state.terrain.yieldRelief = comparison;
      this.state.terrain.valuesId = comparison.values?.dataset_id || null;
    }
    await this.afterTerrainRun(d.id);
    this.renderTab();
  },

  /* ------------------------------------------------------- the findings */

  terrainFindingsPanel(summary) {
    const findings = summary.findings || [];
    const noteClass = { ok: "ok", warning: "warning" };
    // The sentences carry their own numbers, written by the analyser in the
    // units on screen: they are the finding, and converting one here would
    // mean re-deciding what it says. A change of units fetches them again
    // (see refreshPhrasing), so what is drawn is never a unit behind the
    // tables beside it.
    return `
      <div class="panel">
        ${this.printableHead("What the relief says", "btn-print-terrain")}
        ${findings.length ? findings.map((f) => `
          <div class="note ${noteClass[f.level] || ""}" style="margin-bottom:6px">
            ${this.escape(f.text)}</div>`).join("")
          : '<div class="empty">The analysis produced no findings.</div>'}
      </div>`;
  },

  /* ------------------------------------------------------- the field */

  terrainFieldPanel(summary) {
    const elevation = summary.elevation;
    const trend = summary.trend;
    const grid = summary.grid;
    const wet = summary.wetness;
    const lengthUnit = Units.label.length();
    const areaUnit = Units.label.area();
    const length = (v, decimals) => Units.num(Units.convert.length(v), decimals);
    const area = (v) => Units.num(Units.convert.area(v));

    return `
      <div class="panel">
        <h3>The field</h3>
        <div class="note">${this.escape(summary.character?.why || "")}</div>
        <div class="stat-grid" style="margin-top:8px">
          <div class="stat"><div class="k">Relief</div>
            <div class="v">${length(elevation.relief_m, 1)}</div>
            <div class="d">${lengthUnit} · highest to lowest</div></div>
          <div class="stat"><div class="k">Mean slope</div>
            <div class="v">${Units.num(summary.slope.mean_pct, 1)}</div>
            <div class="d">% · 95% under ${Units.num(summary.slope.p95_pct, 1)} %</div></div>
          <div class="stat"><div class="k">Falls to the</div>
            <div class="v" style="font-size:13px">${trend.gradient_pct >= 0.05
              ? this.escape(trend.direction_label) : "nowhere in particular"}</div>
            <div class="d">${trend.gradient_pct >= 0.05
              ? `${Units.num(trend.gradient_pct, 2)} % · ${length(trend.drop_m, 1)} ` +
                `${lengthUnit} across the field`
              : "no consistent fall across the field"}</div></div>
          <div class="stat"><div class="k">Likely wet ground</div>
            <div class="v">${area(wet.wet_area_ha)}</div>
            <div class="d">${areaUnit} · ${Units.num(wet.wet_pct, 1)} % of the field</div></div>
          <div class="stat"><div class="k">Highest / lowest</div>
            <div class="v" style="font-size:13px">${length(elevation.max_m, 1)} /
              ${length(elevation.min_m, 1)}</div>
            <div class="d">${lengthUnit}</div></div>
          <div class="stat"><div class="k">Grid</div>
            <div class="v">${length(grid.cell_m, 1)}</div>
            <div class="d">${lengthUnit} cells · ${area(grid.area_ha)} ${areaUnit}</div></div>
        </div>
        <p class="hint tight">The slope, aspect, landform and wet shares are read
        inside the field's outermost ring of cells — ${area(grid.interior_ha)} of
        the ${area(grid.area_ha)} ${areaUnit}. The ring is drawn on every layer
        but left out of the shares, because its slope and aspect lean on values
        copied from inside the field.</p>
        <h4>Heights across the field</h4>
        <div id="terrain-elevation-histogram"></div>
      </div>`;
  },

  /* ------------------------------------------- yield against the relief */

  /* The layers that could be read against the relief: anything carrying a
   * measured value. An elevation layer is left out because its value IS
   * the height — comparing it with the relief would answer itself — and so
   * are the zone layers this analyser cut, whose values are class codes. */
  terrainValueCandidates() {
    return (this.state.datasets || []).filter((d) => {
      if (d.meta?.extra?.zones_by) return false;
      if (d.meta?.operation === "elevation") return false;
      return (d.columns || []).includes("value");
    });
  },

  /* What the picker opens on: what was compared last, then the file the
   * project calls its yield map, then the analysed file itself when that
   * file is a yield map — which is the common case, one export carrying
   * both the height and the tonnes. */
  terrainValueDefault(analysis, candidates) {
    const ids = new Set(candidates.map((d) => d.id));
    if (analysis.valuesId && ids.has(analysis.valuesId)) return analysis.valuesId;
    const roled = (this.state.project?.layers || [])
      .find((layer) => layer.role === "yield" && ids.has(layer.dataset_id));
    if (roled) return roled.dataset_id;
    if (ids.has(analysis.id)) return analysis.id;
    return candidates[0]?.id || null;
  },

  /* The conversion and the unit this comparison's value takes on screen —
   * bushels for a harvest, pounds for an application. Read off the layer's
   * own operation rather than assumed, or a fertilizer rate would be shown
   * as a grain yield. */
  terrainYieldUnit(comparison) {
    return Units.forColumn(
      comparison.values.column || "value", comparison.values.operation);
  },

  terrainYieldPanel(analysis) {
    const candidates = this.terrainValueCandidates();
    if (!candidates.length) {
      return this.missingPanel(
        "Yield against the relief",
        "The relief says where the water sits and which way the field falls. "
        + "What it costs needs the other half: a yield map, or an as-applied "
        + "log, of the same field. None is open yet.",
        { cta: "load", label: "Open the yield map of this field",
          hint: "It can be the very file whose relief is on screen — one export "
                + "usually carries both the height and the tonnes — or a separate "
                + "file, in which case it need not even share a map projection." });
    }

    const chosen = this.terrainValueDefault(analysis, candidates);
    const comparison = analysis.yieldRelief;
    return `
      <div class="panel">
        <h3>Yield against the relief</h3>
        <p class="hint tight">The same points, read at the height they were
        harvested at: what each band of the field returned, and how much of the
        season's variation the relief explains at all.</p>
        <div class="row tight" style="align-items:flex-end">
          ${this.field("Value layer", this.selectInput("terrain-yield-dataset",
            candidates.map((d) => [d.id, d.label]), chosen))}
          <button class="primary" id="btn-terrain-yield"
                  style="flex:none;margin-bottom:11px">Compare</button>
        </div>
        ${comparison ? this.terrainYieldResult(comparison) : `
          <p class="hint tight">Press Compare and the chart below draws the
          value across the height of the field.</p>`}
      </div>`;
  },

  terrainYieldResult(comparison) {
    const info = this.terrainYieldUnit(comparison);
    const lengthUnit = Units.label.length();
    // Every finding but the one about cleaning: that one is about where the
    // numbers came from rather than about what they say, so it goes at the
    // foot of the section, beside the line naming what was compared.
    const findings = (comparison.findings || [])
      .filter((f) => !f.text.includes("has not been cleaned"));
    const uncleaned = (comparison.findings || [])
      .find((f) => f.text.includes("has not been cleaned"));
    const noteClass = { ok: "ok", warning: "warning" };
    const points = comparison.points;

    return `
      ${findings.map((f) => `
        <div class="note ${noteClass[f.level] || ""}" style="margin-bottom:6px">
          ${this.escape(f.text)}</div>`).join("")}

      <h4>${this.escape(comparison.values.value_label || "Value")} across the
        height of the field</h4>
      <div id="terrain-yield-chart"></div>
      <p class="hint tight">The line is each band's mean and the shading the
      middle half of its readings; the rule is the field average. The strip
      underneath is how much ground the field has at each height — a band at
      either extreme carries less of it, so its mean is a thinner claim.
      Hovering a band gives its range, its mean, its spread and how many
      readings it holds.</p>

      ${this.terrainYieldBandTable(comparison, info, lengthUnit)}
      ${this.terrainYieldTables(comparison, info)}

      <p class="hint tight" style="margin-top:10px">
        ${this.escape(comparison.values.label)}
        (${this.escape(comparison.values.value_label || "value")}, ${this.escape(info.unit)})
        read against the relief of ${this.escape(comparison.terrain.label)}:
        ${Units.num(points.matched, 0)} of ${Units.num(points.total, 0)} readings fell on
        the analysed field${points.off_grid
          ? `, ${Units.num(points.off_grid, 0)} outside it` : ""}.</p>
      ${uncleaned ? `<div class="note warning">${this.escape(uncleaned.text)}</div>` : ""}`;
  },

  /* Every band, in a fold: the chart's own numbers written out, so nothing
   * it shows is reachable only by hovering it. */
  terrainYieldBandTable(comparison, info, lengthUnit) {
    const areaUnit = Units.label.area();
    const length = (v) => Units.num(Units.convert.length(v), 1);
    const rows = (comparison.elevation_bands || []).map((band) => `
      <tr><td>${length(band.from_m)}–${length(band.to_m)}</td>
        <td class="num">${Units.num(Units.convert.area(band.area_ha), 1)}</td>
        <td class="num">${Units.num(info.conv(band.mean), 1)}</td>
        <td class="num">${Units.num(info.conv(band.p25), 1)}–${
          Units.num(info.conv(band.p75), 1)}</td>
        <td class="num">${this.terrainYieldDelta(band.delta_pct)}</td>
        <td class="num">${Units.num(band.points, 0)}</td></tr>`).join("");
    return `
      <details class="fold"><summary>Every band</summary>
        <div class="inner scroll-x"><table class="data">
          <tr><th>Height (${lengthUnit})</th><th>Ground (${areaUnit})</th>
            <th>Mean (${this.escape(info.unit)})</th><th>Middle half</th>
            <th>vs field</th><th>Readings</th></tr>${rows}
        </table></div>
      </details>`;
  },

  /* The three splits the relief offers, each as a small table. The
   * landform classes are sorted by their mean, best first: they have no
   * natural order, so the ranking is the whole point of the table. The
   * slope classes keep the agronomic order — flat to steep is a scale, and
   * reordering it would hide the trend the order itself carries — and the
   * wet ground leads its table, because it is the row being asked about. */
  terrainYieldTables(comparison, info) {
    const landforms = (comparison.landforms || [])
      .filter((cls) => cls.points)
      .slice()
      .sort((a, b) => b.mean - a.mean);
    const slopes = (comparison.slope_classes || []).filter((cls) => cls.points);
    const wetness = comparison.wetness || [];

    const row = (label, cls) => `
      <tr><td>${label}</td>
        <td class="num">${Units.num(info.conv(cls.mean), 1)}</td>
        <td class="num">${this.terrainYieldDelta(cls.delta_pct)}</td>
        <td class="num">${Units.num(cls.points, 0)}</td></tr>`;
    const table = (title, rows, note) => (rows ? `
      <h4>${title}</h4>
      <div class="scroll-x"><table class="data">
        <tr><th>Class</th><th>Mean (${this.escape(info.unit)})</th>
          <th>vs field</th><th>Readings</th></tr>${rows}
      </table></div>
      ${note ? `<p class="hint tight">${note}</p>` : ""}` : "");

    const chip = (color) => `<span style="display:inline-block;width:10px;height:10px;
      border-radius:2px;margin-right:6px;background:${this.escape(color)}"></span>`;

    return `
      ${table("By landform", landforms.map((cls) =>
        row(`${chip(cls.color)}${this.escape(cls.label)}`, cls)).join(""),
        "Best first: the classes have no order of their own, so the ranking is "
        + "what the table is for. The colours are the landform map's.")}
      ${table("By slope class", slopes.map((cls) =>
        row(this.escape(cls.label), cls)).join(""),
        "In the agronomic order, flat to steep, because the order is a scale: "
        + "read down the column and the trend, if there is one, is the shape.")}
      ${table("Wet against well-drained", wetness.filter((w) => w.points).map((w) =>
        row(this.escape(w.label), w)).join(""),
        "The wet ground first. It can cost yield or gain it — in a dry year the "
        + "low ground holds the water the rest of the field wanted — and which "
        + "one it is decides whether drainage is worth pricing.")}`;
  },

  /* A difference from the field average, with its sign written in: a bare
   * "3.2" under "vs field" is read as above average by everyone and as
   * below average by no one. */
  terrainYieldDelta(deltaPct) {
    if (deltaPct == null || !isFinite(deltaPct)) return "—";
    return `${deltaPct > 0 ? "+" : ""}${Units.num(deltaPct, 1)} %`;
  },

  bindTerrainYield() {
    document.getElementById("terrain-yield-dataset")?.addEventListener("change", (event) => {
      const analysis = this.terrainAnalysis();
      if (analysis) analysis.valuesId = event.target.value;
    });
    document.getElementById("btn-terrain-yield")?.addEventListener("click", (event) =>
      this.runTerrainYield(event.currentTarget));
  },

  async runTerrainYield(button) {
    const analysis = this.terrainAnalysis();
    if (!analysis) {
      this.toast("Nothing to compare against",
        "Run the analysis above first; the value is read on its grid.", "warn");
      return;
    }
    const chosen = this.value("terrain-yield-dataset") || analysis.valuesId;
    if (!chosen) {
      this.toast("No value layer", "Open a yield map of this field first.", "warn");
      return;
    }
    const comparison = await this.busy(button, () =>
      this.api(`/api/terrain/${analysis.id}/yield`, {
        method: "POST",
        // The unit set travels with the request because the findings are
        // sentences with their numbers written in: they come back in the
        // units they will be read in, and are shown exactly as they come.
        body: { yield_dataset_id: chosen, units: Units.get() },
      }));
    if (!comparison) return;

    analysis.valuesId = chosen;
    analysis.yieldRelief = comparison;
    // A profile already drawn gains its value series straight away, rather
    // than waiting for the line to be drawn a second time.
    if (analysis.profile?.points) await this.runProfile(analysis.profile.points);
    this.renderTab();

    const strongest = comparison.relations?.strongest;
    const info = this.terrainYieldUnit(comparison);
    this.toast("Read against the relief",
      `${Units.num(comparison.points.matched, 0)} readings, averaging ` +
      `${Units.num(info.conv(comparison.overall.mean), 1)} ${info.unit}. ` +
      (strongest
        ? `${strongest.label[0].toUpperCase()}${strongest.label.slice(1)} explains ` +
          `${Units.num(100 * strongest.r2, 0)} % of the variation.`
        : "The relief varies too little here to explain anything."));
  },

  drawTerrainYieldChart() {
    const analysis = this.terrainAnalysis();
    const comparison = analysis?.yieldRelief;
    const box = document.getElementById("terrain-yield-chart");
    if (!box || !comparison) return;
    const info = this.terrainYieldUnit(comparison);
    const lengthUnit = Units.label.length();
    const areaUnit = Units.label.area();
    const length = (v) => Units.num(Units.convert.length(v), 1);

    const bands = (comparison.elevation_bands || []).map((band) => ({
      from: Units.convert.length(band.from_m),
      to: Units.convert.length(band.to_m),
      at: Units.convert.length(band.mean_elev_m),
      mean: info.conv(band.mean),
      p25: info.conv(band.p25),
      p75: info.conv(band.p75),
      tooltip:
        `${length(band.from_m)}–${length(band.to_m)} ${lengthUnit}\n` +
        `Mean ${Units.num(info.conv(band.mean), 1)} ${info.unit} ` +
        `(${this.terrainYieldDelta(band.delta_pct)} against the field)\n` +
        `Middle half ${Units.num(info.conv(band.p25), 1)}–` +
        `${Units.num(info.conv(band.p75), 1)} ${info.unit}\n` +
        `${Units.num(band.points, 0)} readings over ` +
        `${Units.num(Units.convert.area(band.area_ha), 1)} ${areaUnit}`,
    }));

    Charts.valueAcrossElevation(box, {
      bands,
      fieldMean: info.conv(comparison.overall.mean),
      // The same histogram the field panel draws, on the same axis: it is
      // the ground the bands are cut out of.
      density: this.terrainHistogram(
        analysis.summary.elevation.histogram, Units.convert.length),
    }, {
      valueUnit: info.unit,
      lengthUnit,
      densityLabel: `ground (${areaUnit})`,
    });
  },

  /* The third thing a unit change touches, after the map and the panel.
   *
   * Every sentence the app shows was written by the server with its
   * numbers inside it, in the unit set the server was told to write in.
   * A redraw cannot convert one — converting a sentence would mean
   * re-deciding what it says — so the server is told the new set and
   * asked for the same reports again. Only the prose comes back changed:
   * every number in a table, a legend or a chart is converted here and
   * needs no round trip, and nothing stored moves at all. */
  async sendDisplayUnits() {
    return this.api("/api/units/display", {
      method: "PUT", body: { units: Units.get() },
    }).catch(() => null);
  },

  async refreshPhrasing() {
    await this.sendDisplayUnits();
    const id = this.state.selectedId;
    if (id) {
      // The cleaning and the economic report are cached here as they came
      // off the wire; the tab fetches what it is missing, so dropping them
      // is how they are asked for again.
      delete this.state.reports[`${id}:clean`];
      delete this.state.reports[`${id}:difm`];
      const detail = await this.api(`/api/datasets/${id}`).catch(() => null);
      // The selection may have moved on while the request was out.
      if (detail && this.state.selectedId === id && this.state.selected) {
        Object.assign(this.state.selected, detail);
      }
    }
    await this.refreshTerrainPhrasing();
    this.renderTab();
  },

  async refreshTerrainPhrasing() {
    const analysis = this.terrainAnalysis();
    if (!analysis) return;
    // The relief's own findings first: the summary is the server's, and
    // it says the same numbers in whichever units it is asked for.
    const relief = await this.api(`/api/terrain/${analysis.id}`).catch(() => null);
    if (relief?.summary && this.terrainAnalysis() === analysis) {
      analysis.summary = relief.summary;
    }
    if (analysis.yieldRelief) {
      const comparison = await this.api(`/api/terrain/${analysis.id}/yield`, {
        method: "POST",
        body: {
          yield_dataset_id: analysis.yieldRelief.values.dataset_id,
          units: Units.get(),
        },
      }).catch(() => null);
      // The tab may have moved on while the request was out; rewriting the
      // sentences of an analysis nobody is looking at would be harmless but
      // redrawing over the one on screen would not.
      if (!comparison || this.terrainAnalysis() !== analysis) return;
      analysis.yieldRelief = comparison;
      if (analysis.profile?.points) await this.runProfile(analysis.profile.points);
    }
    if (this.state.tab === "terrain") this.renderTab();
  },

  /* ---------------------------------------------------------- the map */

  terrainMapPanel(analysis) {
    if (!analysis.layers?.length) {
      return `
        <div class="panel">
          <h3>On the map</h3>
          <div class="note warning">${this.escape(analysis.layersGone
            || "The map layers of this analysis are not in memory.")}</div>
          <p class="hint tight">The findings and the tables come from the saved
          summary and are complete; the layer images need the arrays, which the
          app keeps for the four most recent analyses. Press "Analyse the relief
          again" above to bring them back.</p>
        </div>`;
    }
    const layer = this.terrainLayer(analysis);
    const lengthUnit = Units.label.length();
    const interval = analysis.contourInterval ?? analysis.summary.contours.interval_m;
    const level = analysis.summary.elevation.level;

    return `
      <div class="panel">
        <h3>On the map</h3>
        ${this.field("Layer", this.selectInput("terrain-layer",
          analysis.layers.map((l) => [l.key, l.label]), layer.key))}
        <label class="field">
          <span>Opacity <span style="color:var(--text-faint)" id="terrain-opacity-value">${
            Math.round(analysis.opacity * 100)}%</span></span>
          <input type="range" id="terrain-opacity" min="0.1" max="1" step="0.05"
                 value="${analysis.opacity}">
        </label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-hillshade"${analysis.hillshade ? " checked" : ""}
            ${layer.key === "hillshade" ? " disabled" : ""}>
          <span>Blend the hillshade in${layer.key === "hillshade"
            ? " (this layer is the hillshade)" : ""}</span></label>
        <div id="terrain-legend"></div>
        <p class="hint tight">The points are off the map while the relief is on
        it: they are drawn over everything and would hide what is underneath.</p>

        <h4>Contour lines</h4>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-contours"${analysis.contoursOn ? " checked" : ""}
            ${level ? " disabled" : ""}>
          <span>${level ? "This field is level: there is nothing to contour"
            : "Draw the contours over the layer"}</span></label>
        ${level ? "" : `
        <div class="row tight" style="align-items:flex-end">
          ${this.field(`Interval (${lengthUnit})`, this.numberInput("terrain-contour-interval",
            interval == null ? "" : Number(Units.convert.length(interval).toFixed(3)), "any", "0"))}
          <button class="small" id="btn-terrain-contours" style="flex:none;margin-bottom:11px">
            Redraw</button>
        </div>
        <p class="hint tight" id="terrain-contour-status">${interval == null
          ? "No interval was drawn."
          : `Drawn every ${Units.num(Units.convert.length(interval), 2)} ${lengthUnit}` +
            `; every fifth line is heavier and carries its height.`}</p>`}

        <h4>Features</h4>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-features"${analysis.featuresOn ? " checked" : ""}>
          <span>Hills, low ground and closed depressions</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-drainage"${analysis.drainageOn ? " checked" : ""}>
          <span>Drainage lines (${Units.num(Units.convert.length(
            analysis.summary.wetness.drainage_length_m), 0)} ${lengthUnit} in all)</span></label>
      </div>`;
  },

  terrainLayer(analysis) {
    const layers = analysis.layers || [];
    return layers.find((l) => l.key === analysis.layerKey) || layers[0];
  },

  /* The unit a layer's numbers are in, once converted for the screen. The
   * metre layers — elevation, the two positions, the ponding depth — follow
   * the length unit; a percentage, a bearing and a dimensionless index are
   * the same everywhere. Read off the layer's own unit rather than a list of
   * keys, so a layer added to the analyser needs no change here. */
  terrainLayerUnit(layer) {
    if (layer.unit === "m") return Units.label.length();
    if (layer.unit === "deg") return "°";
    return layer.unit || "";
  },

  /* A legend value, with as many decimals as the bar's own span needs: the
   * curvatures run over thousandths and would read as four zeros at the two
   * decimals an elevation wants. */
  terrainLayerValue(layer, value) {
    if (value == null || !isFinite(value)) return "—";
    const metres = layer.unit === "m";
    const convert = metres ? Units.convert.length : (v) => v;
    const legend = layer.legend || {};
    const span = Math.abs(convert(legend.vmax ?? 1) - convert(legend.vmin ?? 0));
    const decimals = span >= 1000 ? 0 : span >= 100 ? 1 : span >= 10 ? 2
      : span >= 1 ? 2 : span >= 0.1 ? 3 : 4;
    return Units.num(convert(value), decimals);
  },

  renderTerrainLegend() {
    const box = document.getElementById("terrain-legend");
    const analysis = this.terrainAnalysis();
    if (!box || !analysis) return;
    const layer = this.terrainLayer(analysis);
    const legend = layer?.legend;
    const unit = this.terrainLayerUnit(layer);
    if (!legend) {
      box.innerHTML = `<div class="empty">Nothing inside the field has a
        ${this.escape(layer?.label || "value")}: the layer draws transparent.</div>`;
      return;
    }

    if (legend.kind === "categorical") {
      // The class shares come from the summary, which counts them inside the
      // ring of edge cells; the legend itself only knows the colours.
      const shares = {};
      for (const cls of analysis.summary.landforms?.classes || []) shares[cls.code] = cls;
      box.innerHTML = `
        <div class="legend" style="margin-top:6px">
          <h4>${this.escape(layer.label)}</h4>
          ${legend.classes.map((cls) => {
            const share = shares[cls.code];
            return `<div class="inline" style="gap:6px;margin-bottom:3px;font-size:12px">
              <span style="flex:none;width:14px;height:14px;border-radius:3px;
                background:${this.escape(cls.color)};border:1px solid var(--border)"></span>
              <span style="flex:1;min-width:0">${this.escape(cls.label)}</span>
              ${share ? `<span style="font:11px var(--mono);color:var(--text-muted)">${
                Units.num(Units.convert.area(share.area_ha))} ${Units.label.area()} ·
                ${Units.num(share.pct, 1)}%</span>` : ""}
            </div>`;
          }).join("")}
        </div>`;
      return;
    }

    // Continuous: the ramp is the palette, and the labels are the stops' own
    // values — on the flow-accumulation ramp they are not evenly spaced, and
    // reading them off the ends would misstate every colour between.
    const stops = legend.stops || [];
    const gradient = stops.map((s) =>
      `${s.color} ${(s.position * 100).toFixed(1)}%`).join(", ");
    const labelled = stops.filter((_, i) => i % 2 === 0 || i === stops.length - 1);
    box.innerHTML = `
      <div class="legend" style="margin-top:6px">
        <h4>${this.escape(layer.label)}${unit ? ` (${this.escape(unit)})` : ""}</h4>
        <div class="ramp" style="background:linear-gradient(90deg, ${gradient})"></div>
        <div style="position:relative;height:14px">
          ${labelled.map((stop) => {
            const percent = (stop.position * 100).toFixed(1);
            return `<span style="position:absolute;left:${percent}%;
              transform:translateX(-${percent}%);font:11px var(--mono);
              color:var(--text-muted);white-space:nowrap">${
                this.terrainLayerValue(layer, stop.value)}</span>`;
          }).join("")}
        </div>
      </div>`;
  },

  bindTerrainMapControls() {
    const analysis = this.terrainAnalysis();
    if (!analysis?.layers?.length) return;

    document.getElementById("terrain-layer").addEventListener("change", (event) => {
      analysis.layerKey = event.target.value;
      this.renderTab();
    });
    const opacity = document.getElementById("terrain-opacity");
    opacity.addEventListener("input", (event) => {
      analysis.opacity = Number(event.target.value);
      document.getElementById("terrain-opacity-value").textContent =
        `${Math.round(analysis.opacity * 100)}%`;
      MapView.setImageOpacity(analysis.opacity);
    });
    document.getElementById("terrain-hillshade").addEventListener("change", (event) => {
      analysis.hillshade = event.target.checked;
      this.showTerrainLayer();
    });

    document.getElementById("terrain-contours")?.addEventListener("change", async (event) => {
      analysis.contoursOn = event.target.checked;
      if (analysis.contoursOn && !analysis.contours) await this.loadTerrainContours(null);
      this.drawTerrainContours();
    });
    document.getElementById("btn-terrain-contours")?.addEventListener("click", async (event) => {
      const typed = this.number("terrain-contour-interval", null);
      await this.busy(event.currentTarget, () =>
        this.loadTerrainContours(typed == null ? null : Units.toInternal.length(typed)));
      analysis.contoursOn = true;
      this.renderTab();
    });

    document.getElementById("terrain-features").addEventListener("change", (event) => {
      analysis.featuresOn = event.target.checked;
      this.drawTerrainFeatures();
    });
    document.getElementById("terrain-drainage").addEventListener("change", (event) => {
      analysis.drainageOn = event.target.checked;
      this.drawTerrainFeatures();
    });
  },

  /* The relief on the map: one image, swapped in place, with the contours
   * and the outlines above it. */
  showTerrainLayer() {
    const analysis = this.terrainAnalysis();
    const layer = analysis && this.terrainLayer(analysis);
    if (!layer) return;
    analysis.layerKey = layer.key;

    // The point canvas is drawn over the map rather than in it, so the
    // points would cover the relief entirely. The panel says so beside the
    // picker, and leaving the tab brings them back.
    MapView.clearPoints();
    document.getElementById("legend").hidden = true;

    const query = new URLSearchParams({ run: String(analysis.stamp) });
    if (analysis.hillshade && layer.key !== "hillshade") query.set("hillshade", "1");
    MapView.setImage(`${layer.url}?${query}`, layer.bounds, {
      opacity: analysis.opacity,
      onError: () => this.terrainImageFailed(analysis.id),
    });
    this.state.terrainOnMap = true;

    document.getElementById("map-status").textContent =
      `${layer.label} · ${Units.num(Units.convert.area(analysis.summary.grid.area_ha))} ` +
      `${Units.label.area()}`;
    if (!analysis.fitted) {
      analysis.fitted = true;
      MapView.fit(analysis.summary.grid.bounds_lonlat);
    }
    this.renderTerrainLegend();
    this.drawTerrainContours();
    this.drawTerrainFeatures();
    this.drawProfileLine(this.state.terrainDraw?.points || this.state.terrain?.profile?.points);
  },

  /* The layer images are drawn from the arrays the app keeps for the four
   * most recent analyses, and a fifth analysis takes this one's away. The
   * summary and the tables live on — they are saved with the dataset — so
   * the first sign of the loss is an image that will not load. Asking
   * /layers turns that into the analyser's own sentence about what to do,
   * and a transient failure that /layers survives changes nothing. */
  async terrainImageFailed(id) {
    const analysis = this.terrainAnalysis();
    if (!analysis || analysis.id !== id || analysis.layersGone) return;
    try {
      await this.api(`/api/terrain/${id}/layers`);
      return;
    } catch (err) {
      analysis.layers = null;
      analysis.layersGone = err.message;
    }
    this.leaveTerrainMap();
    this.renderTab();
  },

  /* Everything this tab put on the map comes off when the tab is left: the
   * relief belongs to this panel, and the points it replaced come back. */
  leaveTerrainMap() {
    if (!this.state.terrainOnMap) return;
    this.state.terrainOnMap = false;
    this.endProfile();
    MapView.clearImage();
    for (const name of this.TERRAIN_LAYERS) MapView.clearGeoJson(name);
    this.loadMap();
  },

  /* ------------------------------------------------------------ contours */

  async loadTerrainContours(intervalM) {
    const analysis = this.terrainAnalysis();
    if (!analysis) return null;
    const query = intervalM == null ? "" : `?interval_m=${encodeURIComponent(intervalM)}`;
    const collection = await this.api(
      `/api/terrain/${analysis.id}/contours${query}`).catch((err) => {
        this.toast("The contours did not come", err.message, "error");
        return null;
      });
    if (!collection) return null;
    analysis.contours = collection;
    // What was asked for and what was drawn are not always the same: a level
    // field ignores the request, so the box shows the interval actually used.
    analysis.contourInterval = collection.interval_m;
    return collection;
  },

  drawTerrainContours() {
    const analysis = this.terrainAnalysis();
    if (!analysis?.contoursOn || !analysis.contours) {
      MapView.clearGeoJson("terrain-contours");
      return;
    }
    const lengthUnit = Units.label.length();
    MapView.setGeoJson("terrain-contours", analysis.contours, {
      style: (p) => ({
        color: "#3d3222", weight: p.major ? 2 : 1, opacity: p.major ? 0.95 : 0.7,
      }),
      // Every fifth line carries its height where it is drawn; the rest
      // answer on hover, or the labels would cover the layer they describe.
      tooltip: (p) => ({
        text: `${Units.num(Units.convert.length(p.level_m), 1)} ${lengthUnit}`,
        permanent: !!p.major,
        className: "contour-label",
      }),
    });
  },

  /* ------------------------------------------------------------ features */

  terrainFeaturesPanel(summary) {
    const features = summary.features;
    const lengthUnit = Units.label.length();
    const areaUnit = Units.label.area();
    const length = (v, decimals = 2) => Units.num(Units.convert.length(v), decimals);
    const area = (v) => Units.num(Units.convert.area(v));
    const none = !features.hills.length && !features.lows.length && !features.depressions.length;

    const table = (title, header, rows) => rows.length ? `
      <h4>${title}</h4>
      <div class="scroll-x"><table class="data">
        <tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>
        ${rows.join("")}
      </table></div>` : "";

    // Five columns each: what it is, where, how big, how far it stands out,
    // and the one thing that follows from it. The absolute heights — the
    // summit, the bottom, the spill point — are in the popup on the map and
    // on the printed page; in a panel this wide they would push the share
    // and the ponding off the edge, and they are the numbers a farmer reads
    // last.
    const hills = features.hills.map((h) => `
      <tr class="clickable" data-feature="${h.lon},${h.lat}">
        <td>${this.escape(h.label)}</td><td>${this.escape(h.position)}</td>
        <td class="num">${length(h.height_m, 1)}</td>
        <td class="num">${area(h.area_ha)}</td>
        <td class="num">${Units.num(h.mean_slope_pct, 1)}</td></tr>`);
    const lows = features.lows.map((l) => `
      <tr class="clickable" data-feature="${l.lon},${l.lat}">
        <td>${this.escape(l.label)}</td><td>${this.escape(l.position)}</td>
        <td class="num">${length(l.depth_m, 1)}</td>
        <td class="num">${area(l.area_ha)}</td>
        <td>${l.closed ? "ponds" : "runs off"}</td></tr>`);
    const depressions = features.depressions.map((d) => `
      <tr class="clickable" data-feature="${d.lon},${d.lat}">
        <td>${this.escape(d.label)}</td><td>${this.escape(d.position)}</td>
        <td class="num">${area(d.area_ha)}</td>
        <td class="num">${length(d.max_depth_m, 2)}</td>
        <td class="num">${Units.num(this.terrainVolume(d.volume_m3), 0)}</td></tr>`);

    return `
      <div class="panel">
        <h3>Hills, low ground and depressions</h3>
        ${none ? `<div class="note">No hill, hollow or closed depression stands
          more than ${length(summary.features.depression_floor_m, 2)} ${lengthUnit}
          out of the general fall of this field.</div>` : ""}
        ${table("Hills", ["Hill", "Where", `Rises (${lengthUnit})`, `Area (${areaUnit})`,
                          "Slope %"], hills)}
        ${table("Low ground", ["Low", "Where", `Below (${lengthUnit})`,
                               `Area (${areaUnit})`, "Water"], lows)}
        ${table("Closed depressions", ["Depression", "Where", `Area (${areaUnit})`,
                                       `Depth (${lengthUnit})`,
                                       `Holds (${this.terrainVolumeUnit()})`], depressions)}
        ${features.depressions_unlisted ? `<p class="hint tight">
          ${features.depressions_unlisted} shallower hollow(s) were not listed: they
          are under the ${length(features.depression_floor_m, 2)} ${lengthUnit} a
          feature has to stand out by, which is twice the noise in these readings.
          Lower "smallest feature" in the options above to list them, and read what
          comes out knowing it is close to the noise.</p>` : ""}
        ${none ? "" : `<p class="hint tight">A row centres the map on its feature.
          The outlines carry the same numbers in a popup.</p>`}
      </div>`;
  },

  /* A volume in the user's length unit cubed. There is no volume group in
   * the unit catalogue — nothing else in the app measures one — so the
   * length factor is cubed here: a cubic foot is 0.3048³ of a cubic metre.
   * The unit is written beside every number, because a pond volume is the
   * one figure nobody can check by eye. */
  terrainVolume(cubicMetres) {
    if (cubicMetres == null || !isFinite(cubicMetres)) return null;
    const factor = Units.factor("length", Units.label.length());
    return cubicMetres / (factor * factor * factor);
  },

  terrainVolumeUnit() { return `${Units.label.length()}³`; },

  drawTerrainFeatures() {
    const analysis = this.terrainAnalysis();
    const collection = analysis?.features;
    if (!analysis) return;

    if (!analysis.featuresOn || !collection) {
      MapView.clearGeoJson("terrain-features");
    } else {
      const lengthUnit = Units.label.length();
      const areaUnit = Units.label.area();
      const length = (v, d = 1) => `${Units.num(Units.convert.length(v), d)} ${lengthUnit}`;
      const area = (v) => `${Units.num(Units.convert.area(v))} ${areaUnit}`;
      MapView.setGeoJson("terrain-features", collection, {
        style: (p) => {
          const style = this.TERRAIN_FEATURE_STYLE[p.kind] || this.TERRAIN_FEATURE_STYLE.hill;
          return {
            color: style.color, weight: 2, dashArray: style.dashArray,
            fillColor: style.color, fillOpacity: p.kind === "hill" ? 0.12 : 0.22,
          };
        },
        popup: (p) => {
          const lines = [`<b>${this.escape(p.label)}</b> — ${this.escape(p.position)}`];
          if (p.kind === "hill") {
            lines.push(`Rises ${length(p.height_m)} above its surroundings`,
                       `${area(p.area_ha)} · top at ${length(p.summit_m)}`,
                       `Averages ${Units.num(p.mean_slope_pct, 1)} % slope`);
          } else if (p.kind === "low") {
            lines.push(`Lies ${length(p.depth_m)} below the ground around it`,
                       `${area(p.area_ha)} · bottom at ${length(p.bottom_m)}`,
                       p.closed ? "Part of it is closed: water ponds there"
                                : "Open: the water runs out");
          } else {
            lines.push(`${area(p.area_ha)} · ${length(p.max_depth_m, 2)} at its deepest`,
                       `Holds ${Units.num(this.terrainVolume(p.volume_m3), 0)} ` +
                       `${this.terrainVolumeUnit()} before it spills`,
                       `Spills at ${length(p.spill_m)}`);
          }
          return lines.join("<br>");
        },
        tooltip: (p) => this.escape(p.label),
      });
    }

    if (!analysis.drainageOn || !collection?.drainage) {
      MapView.clearGeoJson("terrain-drainage");
      return;
    }
    MapView.setGeoJson("terrain-drainage", collection.drainage, {
      style: () => ({ color: "#2b6cb0", weight: 2, opacity: 0.85, dashArray: "6 4" }),
      tooltip: (p) => `${Units.num(Units.convert.length(p.length_m), 0)} ` +
        `${Units.label.length()} of drainage line`,
    });
  },

  /* ------------------------------------------------------------- profile */

  terrainProfilePanel() {
    const analysis = this.terrainAnalysis();
    const profile = analysis?.profile;
    const lengthUnit = Units.label.length();
    const drawing = !!this.state.terrainDraw;
    const length = (v, d = 1) => Units.num(Units.convert.length(v), d);

    let numbers = "";
    if (profile) {
      const measured = profile.elev_m.filter((z) => z != null);
      const fall = measured.length ? Math.max(...measured) - Math.min(...measured) : null;
      const steps = profile.slope_pct.filter((s) => s != null);
      const steepest = steps.length
        ? steps.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a)) : null;
      const rows = profile.distance_m.map((distance, i) => `
        <tr><td class="num">${length(distance, 0)}</td>
          <td class="num">${profile.elev_m[i] == null ? "off the field"
            : length(profile.elev_m[i], 1)}</td>
          <td class="num">${profile.slope_pct[i] == null ? "—"
            : Units.num(profile.slope_pct[i], 2)}</td></tr>`).join("");
      numbers = `
        <div class="stat-grid" style="margin-top:8px">
          <div class="stat"><div class="k">Length</div>
            <div class="v">${length(profile.length_m, 0)}</div>
            <div class="d">${lengthUnit}</div></div>
          <div class="stat"><div class="k">Fall along the line</div>
            <div class="v">${length(fall, 1)}</div>
            <div class="d">${lengthUnit} · highest to lowest</div></div>
        </div>
        <p class="hint tight">Steepest step ${steepest == null ? "—"
          : `${Units.num(Math.abs(steepest), 2)} %, ${steepest > 0 ? "downhill" : "uphill"}`}
          along the walk. Downhill is positive.</p>
        <details class="fold"><summary>Every station</summary>
          <div class="inner scroll-x"><table class="data">
            <tr><th>Distance (${lengthUnit})</th><th>Height (${lengthUnit})</th>
              <th>Slope from the last (%)</th></tr>${rows}
          </table></div>
        </details>`;
    }

    return `
      <div class="panel">
        <h3>Profile along a line</h3>
        <div class="row tight">
          <button id="btn-terrain-profile" class="${drawing ? "" : "primary"}">
            ${drawing ? "Drawing…" : "Draw a profile"}</button>
          <button id="btn-terrain-profile-done"${drawing ? "" : " disabled"}>Done</button>
        </div>
        <p class="hint tight" id="terrain-profile-status">${drawing
          ? "Click the start and the end on the map. Each further click extends the "
            + "line. Escape or Done ends it; pressing Draw again starts a new one."
          : "Click two points on the map and the ground between them is drawn "
            + "below, height against distance."}</p>
        <div id="terrain-profile-chart"></div>
        ${profile?.values_meta ? `<p class="hint tight">${
          this.escape(profile.values_meta.note)} The two panels share the
          distance axis rather than sharing a plot: a height and a yield have
          no common scale, so where two lines drawn against two scales crossed
          would be decided by the axes, not by the field.</p>` : ""}
        ${numbers}
      </div>`;
  },

  bindTerrainPanels() {
    document.getElementById("btn-terrain-profile")
      ?.addEventListener("click", () => this.startProfile());
    document.getElementById("btn-terrain-profile-done")
      ?.addEventListener("click", () => { this.endProfile(); this.renderTab(); });

    for (const row of document.querySelectorAll("[data-feature]")) {
      row.addEventListener("click", () => {
        const [lon, lat] = row.dataset.feature.split(",").map(Number);
        MapView.panTo(lon, lat, 16);
      });
    }

    document.getElementById("btn-terrain-zones")
      ?.addEventListener("click", (event) => this.makeTerrainZones(event.currentTarget));
    document.getElementById("btn-terrain-export")
      ?.addEventListener("click", (event) => this.runTerrainExport(event.currentTarget));
  },

  startProfile() {
    const analysis = this.terrainAnalysis();
    if (!analysis) {
      this.toast("Nothing to profile",
        "Run the analysis above and the line can be drawn over the result.", "warn");
      return;
    }
    // A second press starts a new line rather than adding to the old one.
    this.endProfile();
    const draw = { points: [] };
    draw.onMapClick = (event) => {
      draw.points.push([event.latlng.lng, event.latlng.lat]);
      this.drawProfileLine(draw.points);
      if (draw.points.length >= 2) this.runProfile(draw.points);
      else this.setProfileStatus("Start set. Click where the line should end.");
    };
    draw.onKey = (event) => { if (event.key === "Escape") { this.endProfile(); this.renderTab(); } };
    // The outlines and the contours stay on the map but stop answering
    // clicks, or a line drawn across a hill would open the hill's popup
    // instead of setting its second point.
    MapView.setClickThrough(true);
    MapView.onClick(draw.onMapClick);
    document.addEventListener("keydown", draw.onKey);
    this.state.terrainDraw = draw;
    MapView.clearGeoJson("terrain-profile");
    this.renderTab();
  },

  endProfile() {
    const draw = this.state.terrainDraw;
    if (!draw) return;
    MapView.offClick(draw.onMapClick);
    document.removeEventListener("keydown", draw.onKey);
    MapView.setClickThrough(false);
    this.state.terrainDraw = null;
  },

  setProfileStatus(text) {
    const node = document.getElementById("terrain-profile-status");
    if (node) node.textContent = text;
  },

  /* The line as it is drawn, so the click is answered before the profile
   * comes back from the server. */
  drawProfileLine(points) {
    if (!points?.length) { MapView.clearGeoJson("terrain-profile"); return; }
    const features = points.map((point, i) => ({
      type: "Feature", properties: { station: i + 1 },
      geometry: { type: "Point", coordinates: point },
    }));
    if (points.length >= 2) {
      features.push({
        type: "Feature", properties: { station: 0 },
        geometry: { type: "LineString", coordinates: points },
      });
    }
    MapView.setGeoJson("terrain-profile", { type: "FeatureCollection", features }, {
      style: (p) => (p.station
        ? { radius: 4, color: "#ffffff", weight: 2, fillColor: "#b33a3a", fillOpacity: 1 }
        : { color: "#b33a3a", weight: 3, opacity: 0.9 }),
    });
  },

  async runProfile(points) {
    const analysis = this.terrainAnalysis();
    if (!analysis) return;
    // Two hundred stations read every cell a line crosses on any field this
    // app grids, and a chart a panel wide has fewer pixels than that.
    // The value layer rides along once it has been compared, so the chart
    // under the map gains its second panel without a second click. Before
    // that the request is the one it has always been.
    const body = { points, n: 200 };
    if (analysis.yieldRelief) {
      body.values_dataset_id = analysis.yieldRelief.values.dataset_id;
      body.units = Units.get();
    }
    const profile = await this.api(`/api/terrain/${analysis.id}/profile`, {
      method: "POST", body,
    }).catch((err) => {
      this.setProfileStatus(err.message);
      return null;
    });
    if (!profile) return;
    analysis.profile = profile;
    this.renderTab();
    this.setProfileStatus(
      `${points.length} point(s) on the line. Click again to extend it, `
      + "Escape or Done to finish.");
  },

  /* --------------------------------------------------- slope and aspect */

  terrainSlopePanel(summary) {
    const areaUnit = Units.label.area();
    const rows = summary.slope.classes.map((cls) => `
      <tr><td>${this.escape(cls.label)}</td>
        <td class="num">${Units.num(Units.convert.area(cls.area_ha))}</td>
        <td class="num">${Units.num(cls.pct, 1)}</td></tr>`).join("");
    return `
      <div class="panel">
        <h3>Slope</h3>
        <div id="terrain-slope-histogram"></div>
        <p class="hint tight">The dashed lines are the agronomic breaks: under
        2 % the water leaves slowly, 2–5 % is where sheet erosion starts on bare
        soil, and above 10 % both the machinery and the erosion become a
        concern.</p>
        <div class="scroll-x"><table class="data">
          <tr><th>Class</th><th>Area (${areaUnit})</th><th>Share (%)</th></tr>${rows}
        </table></div>
      </div>`;
  },

  terrainAspectPanel(summary) {
    const areaUnit = Units.label.area();
    const sectors = summary.aspect.sectors || [];
    const flat = sectors.find((s) => s.key === "flat");
    const rows = sectors.map((sector) => `
      <tr><td>${this.escape(sector.label)}</td>
        <td class="num">${Units.num(Units.convert.area(sector.area_ha))}</td>
        <td class="num">${Units.num(sector.pct, 1)}</td></tr>`).join("");
    return `
      <div class="panel">
        <h3>Which way it faces</h3>
        <div id="terrain-aspect-rose"></div>
        <p class="hint tight">${flat && flat.pct > 0
          ? `${Units.num(flat.pct, 1)} % of the field (${Units.num(
              Units.convert.area(flat.area_ha))} ${areaUnit}) is too flat to face
              anywhere, and is not drawn as a petal.`
          : "Every part of this field falls one way or another."}
          A south-facing slope dries and warms first in the spring.</p>
        <details class="fold"><summary>Every sector</summary>
          <div class="inner scroll-x"><table class="data">
            <tr><th>Facing</th><th>Area (${areaUnit})</th><th>Share (%)</th></tr>${rows}
          </table></div>
        </details>
      </div>`;
  },

  terrainLandformPanel(summary) {
    const areaUnit = Units.label.area();
    const classes = summary.landforms?.classes || [];
    if (!classes.length) return "";
    const lengthUnit = Units.label.length();
    return `
      <div class="panel">
        <h3>Landform</h3>
        <div class="scroll-x"><table class="data">
          <tr><th>Class</th><th>Area (${areaUnit})</th><th>Share (%)</th></tr>
          ${classes.map((cls) => `
            <tr><td><span style="display:inline-block;width:10px;height:10px;
              border-radius:2px;margin-right:6px;background:${this.escape(cls.color)}"></span>${
              this.escape(cls.label)}</td>
              <td class="num">${Units.num(Units.convert.area(cls.area_ha))}</td>
              <td class="num">${Units.num(cls.pct, 1)}</td></tr>`).join("")}
        </table></div>
        <p class="hint tight">Each cell is placed by how high it sits against the
        ground within ${Units.num(Units.convert.length(summary.landforms.tpi_small_m), 0)}
        and ${Units.num(Units.convert.length(summary.landforms.tpi_large_m), 0)}
        ${lengthUnit} of it — the near view says whether it is on a crest or in a
        hollow, the far view where that sits in the field as a whole.</p>
      </div>`;
  },

  drawTerrainCharts() {
    const analysis = this.terrainAnalysis();
    if (!analysis) return;
    const summary = analysis.summary;
    const lengthUnit = Units.label.length();

    const elevation = document.getElementById("terrain-elevation-histogram");
    if (elevation) {
      Charts.histogram(elevation, null, this.terrainHistogram(
        summary.elevation.histogram, Units.convert.length), {
        height: 120,
        format: (v) => `${Units.num(v, 0)} ${lengthUnit}`,
      });
    }

    const slope = document.getElementById("terrain-slope-histogram");
    if (slope) {
      Charts.histogram(slope, null, summary.slope.histogram, {
        height: 130,
        format: (v) => `${Units.num(v, 1)} %`,
        marks: summary.slope.classes
          .filter((cls) => cls.from_pct > 0)
          .map((cls) => ({ value: cls.from_pct, label: `${cls.from_pct}%` })),
      });
    }

    const rose = document.getElementById("terrain-aspect-rose");
    if (rose) Charts.rose(rose, summary.aspect.sectors);

    const chart = document.getElementById("terrain-profile-chart");
    if (chart && analysis.profile) {
      const meta = analysis.profile.values_meta;
      const value = meta
        ? Units.forColumn(meta.column || "value", meta.operation)
        : null;
      Charts.profile(chart, {
        distance: analysis.profile.distance_m.map(Units.convert.length),
        elevation: analysis.profile.elev_m.map(
          (z) => (z == null ? null : Units.convert.length(z))),
        values: value && analysis.profile.values
          ? analysis.profile.values.map((v) => (v == null ? null : value.conv(v)))
          : null,
      }, {
        distanceUnit: lengthUnit,
        elevationUnit: lengthUnit,
        elevationLabel: "Height",
        valueUnit: value?.unit,
        valueLabel: meta?.value_label || "Value",
      });
    }

    this.drawTerrainYieldChart();
  },

  /* A histogram's edges in the displayed unit; the counts are counts. */
  terrainHistogram(histogram, convert) {
    if (!histogram?.counts?.length) return histogram;
    return { counts: histogram.counts, edges: histogram.edges.map(convert) };
  },

  /* --------------------------------------------------------------- zones */

  terrainZonesPanel(analysis) {
    const zones = analysis.zones;
    return `
      <div class="panel">
        <h3>Make zones from this</h3>
        <p class="hint tight">A zone layer is an ordinary dataset: it shows up in
        "Loaded data", exports as shapefile, GeoJSON or GeoPackage, and goes to
        the monitor like any other map. This is the bridge from the relief to the
        machine.</p>
        ${this.field("Zones by", this.selectInput("terrain-zones-by", [
          ["landform", "Landform — hilltop, slope, flat, hollow"],
          ["slope_class", "Slope class — the agronomic breaks"],
          ["elevation_bands", "Elevation bands"],
          ["wetness", "Wetness — the wet ground against the rest"],
        ], this.state.terrainZonesBy || "landform"))}
        <div class="row tight">
          ${this.field("As", this.selectInput("terrain-zones-kind", [
            ["polygons", "Polygons — one per zone"],
            ["points", "Points — one per grid cell"],
          ], this.state.terrainZonesKind || "polygons"))}
          ${this.field("Bands", this.numberInput("terrain-zones-bands", 4, "1", "2"),
            "Elevation bands only.")}
        </div>
        <button class="primary wide" id="btn-terrain-zones">Make the zone layer</button>
        <div id="terrain-zones-result">${zones ? this.terrainZonesResult(zones) : ""}</div>
        <p class="hint tight">A zone layer is a map of class codes, not of
        product: give it a rate per zone before it goes to the terminal, or the
        monitor applies 1, 2 and 3. And to fit one response curve per zone, the
        zone column has to sit on the yield layer itself — the economics reads
        its zones from that file's own columns, not from a layer beside it.</p>
      </div>`;
  },

  terrainZonesResult(zones) {
    const dataset = zones.dataset;
    const labels = Object.entries(zones.zone_labels || {})
      .map(([code, label]) => `${code} · ${label}`).join(", ");
    return `
      <div class="note ok" style="margin-top:8px">
        <b>${this.escape(dataset.label)}</b><br>
        ${Units.num(dataset.rows, 0)} ${zones.kind === "points" ? "cells" : "zone(s)"},
        ${Units.num(Units.convert.area(dataset.area_ha))} ${Units.label.area()}.<br>
        <span class="d">${this.escape(labels)}</span>
      </div>
      <div class="row tight" style="margin-top:8px">
        <button class="small" data-cta="inspect:${dataset.id}">Show the new layer</button>
        <button class="small" data-cta="export-file:${dataset.id}">Take it to Export</button>
      </div>`;
  },

  async makeTerrainZones(button) {
    const analysis = this.terrainAnalysis();
    if (!analysis) {
      this.toast("Nothing to cut into zones",
        "Run the analysis above first; the zones are cut from its layers.", "warn");
      return;
    }
    this.state.terrainZonesBy = this.value("terrain-zones-by");
    this.state.terrainZonesKind = this.value("terrain-zones-kind");
    const body = {
      by: this.state.terrainZonesBy,
      kind: this.state.terrainZonesKind,
      bands: this.number("terrain-zones-bands", 4),
    };
    const zones = await this.busy(button, () =>
      this.api(`/api/terrain/${analysis.id}/zones`, { method: "POST", body }));
    if (!zones) return;

    analysis.zones = zones;
    await this.refreshDatasets();
    const box = document.getElementById("terrain-zones-result");
    if (box) {
      box.innerHTML = this.terrainZonesResult(zones);
      this.bindPanelActions(box);
    }
    this.toast("Zone layer made",
      `${zones.dataset.label}: ${Units.num(zones.dataset.rows, 0)} ` +
      `${zones.kind === "points" ? "cells" : "zones"}. It is in "Loaded data", ` +
      "ready to export to the monitor.");
  },

  /* -------------------------------------------------------------- export */

  terrainExportPanel() {
    const analysis = this.terrainAnalysis();
    return `
      <div class="panel">
        <h3>Export the relief</h3>
        <p class="hint tight">One zip for QGIS: the layers as GeoTIFFs on the
        analysis grid, the vectors in WGS84, and a README naming every file and
        the landform codes.</p>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-export-geotiff" checked>
          <span>Layers as GeoTIFF — elevation, slope, aspect, wetness, hillshade,
            landform, ponding depth</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-export-contours" checked>
          <span>Contour lines</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-export-features" checked>
          <span>Hills, low ground and depressions</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="terrain-export-drainage" checked>
          <span>Drainage lines</span></label>
        ${this.field("Vector format", this.selectInput("terrain-export-format", [
          ["shapefile", "Shapefile (.shp)"],
          ["geojson", "GeoJSON"],
        ], this.state.terrainExportFormat || "shapefile"))}
        <button class="primary wide" id="btn-terrain-export">Generate the files</button>
        <div id="terrain-export-report">${
          analysis?.exported ? this.terrainExportReport(analysis.exported) : ""}</div>
      </div>`;
  },

  terrainExportReport(result) {
    return `
      <div style="margin-top:10px">
        <a class="btn primary wide" style="display:block;text-decoration:none"
           href="${result.download_url}" download>
          Download ${this.escape(result.filename)}</a>
        <p class="hint tight">Written to:<br>
          <code style="font-size:11px">${this.escape(result.path)}</code></p>
        <div class="scroll-x"><table class="data">
          <tr><th>File</th></tr>
          ${result.files.map((name) =>
            `<tr><td style="font-family:var(--mono);font-size:11px">${
              this.escape(name)}</td></tr>`).join("")}
        </table></div>
        ${(result.notes || []).map((note) =>
          `<div class="note" style="margin-top:8px">${this.escape(note)}</div>`).join("")}
      </div>`;
  },

  async runTerrainExport(button) {
    const analysis = this.terrainAnalysis();
    if (!analysis) {
      this.toast("Nothing to export",
        "Run the analysis above; the export is written from its layers.", "warn");
      return;
    }
    const include = [
      ["geotiff", this.checked("terrain-export-geotiff")],
      ["contours", this.checked("terrain-export-contours")],
      ["features", this.checked("terrain-export-features")],
      ["drainage", this.checked("terrain-export-drainage")],
    ].filter(([, on]) => on).map(([part]) => part);
    if (!include.length) {
      this.toast("Nothing chosen", "Tick at least one of the four.", "warn");
      return;
    }
    this.state.terrainExportFormat = this.value("terrain-export-format");
    const result = await this.busy(button, () =>
      this.api(`/api/terrain/${analysis.id}/export`, {
        method: "POST",
        body: { include, vector_format: this.state.terrainExportFormat },
      }));
    if (!result) return;

    analysis.exported = result;
    const box = document.getElementById("terrain-export-report");
    if (box) box.innerHTML = this.terrainExportReport(result);
    this.toast("Terrain export written",
      `${result.filename}: ${result.files.length} file(s).\nSaved at ${result.path}`);
  },

});

/* ======================================================================
 * Cleaning
 * ==================================================================== */

Object.assign(App, {
  tabLimpeza(panel) {
    const d = this.state.selected;
    if (!d) {
      const loaded = this.state.datasets.length;
      panel.innerHTML = loaded
        ? this.missingPanel(
            "Cleaning",
            "Cleaning works on one file at a time, and none is picked yet.",
            { cta: "pick", label: "Show the first file" })
        : this.missingPanel(
            "Cleaning",
            "Nothing is open yet. This tab removes overlap, headland turns and "
            + "sensor faults from a monitor file, and says what it took out and "
            + "from where. Nothing is overwritten: it produces a clean copy and "
            + "the removed records beside the original.",
            { cta: "load", label: "Load a file",
              hint: "Already cleaned elsewhere? You can skip this tab entirely and "
                    + "go straight to Economics." });
      return;
    }

    const presets = this.state.catalog.presets;
    const suggested = presets[d.meta.operation] ? d.meta.operation : "minimal";
    const current = this.state.cleanPreset || suggested;
    const config = presets[current];

    panel.innerHTML = `
      <div class="panel">
        <h3>Data cleaning</h3>
        ${this.field("Profile", this.selectInput("clean-preset",
          Object.entries(presets).map(([k, v]) => [k, v.label]), current))}
        <div class="note">${this.escape(config.description)}</div>
        ${this.field("Variable to clean", this.selectInput("clean-column",
          this.columnOptions(), this.state.colorColumn))}
        ${this.field("Sensor flow delay (s)",
          this.numberInput("clean-delay", config.corrections.flow_delay_s ?? 0, "0.5", "0"),
          "Seconds between the cut and the sensor reading. Zero turns the correction off.")}
        ${this.machinePicker("clean")}
      </div>

      <div class="panel">
        <h3>Filters</h3>
        <div id="clean-steps"></div>
      </div>

      <div class="panel">
        <button class="primary wide" id="btn-run-clean">Run cleaning</button>
        <p class="hint tight">The original data is kept: cleaning creates new
        datasets, "clean" and "removed".</p>
      </div>

      <div id="clean-report"></div>

      ${this.producesPanel([
        { label: "Export the clean copy as a file", cta: "export-clean",
          hint: "The kept records, written as shapefile, CSV or GeoJSON — the file "
                + "to hand on to whatever comes next." },
        { label: "Print the cleaning report (PDF)", cta: "print-clean",
          hint: "Before and after, what each filter removed, and the histogram of "
                + "both — the evidence that the cleaning was sound." },
      ], "Cleaning a file and stopping there is a complete use of the app.")}`;

    this.renderCleanSteps(config.steps);
    this.bindMachinePicker("clean");

    document.getElementById("clean-preset").addEventListener("change", (event) => {
      this.state.cleanPreset = event.target.value;
      this.renderTab();
    });
    document.getElementById("btn-run-clean").addEventListener("click", () => this.runClean());

    const existing = this.state.reports[`${this.state.selectedId}:clean`];
    if (existing) this.renderCleanReport(existing);
    else if (d.has_clean_report) this.restoreCleanReport(d);
  },

  renderCleanSteps(stepConfig) {
    const box = document.getElementById("clean-steps");
    box.innerHTML = "";
    const lengthUnit = Units.label.length();
    const speedUnit = Units.label.speed();
    const { unit: rateUnit, kind: rateKind } =
      Units.forColumn("value", this.state.selected?.meta?.operation);

    /* Parameters that are physical quantities show up in the user's chosen unit
     * and go back to metric on the way out. */
    const PARAM_META = {
      min_fraction: { label: "Minimum fraction of full width", step: "0.05", suffix: "0-1" },
      max_overlap_pct: { label: "Overlap tolerated", step: "5", suffix: "%" },
      buffer_m: { label: "Field edge", step: "1", suffix: lengthUnit, kind: "length" },
      start_m: { label: "Drop at pass start", step: "1", suffix: lengthUnit, kind: "length" },
      end_m: { label: "Drop at pass end", step: "1", suffix: lengthUnit, kind: "length" },
      min_points: { label: "Minimum records", step: "1", suffix: "rec." },
      max_change_pct: { label: "Maximum change", step: "5", suffix: "%" },
      max_jump_m: { label: "Maximum jump", step: "1", suffix: lengthUnit, kind: "length" },
      k: { label: "Standard deviations (k)", step: "0.25", suffix: "sd" },
      lower_pct: { label: "Lower percentile", step: "0.5", suffix: "%" },
      upper_pct: { label: "Upper percentile", step: "0.5", suffix: "%" },
      k_neighbors: { label: "Neighbours", step: "1", suffix: "points" },
      threshold: { label: "Deviation limit", step: "0.5", suffix: "MAD" },
      method: { label: "Method", choices: [["std", "Standard deviation"], ["percentile", "Percentile"]] },
      drop_zero: { label: "Drop zeros", bool: true },
      drop_negative: { label: "Drop negatives", bool: true },
      drop_duplicates: { label: "Drop duplicates", bool: true },
    };

    for (const step of this.state.catalog.steps) {
      const settings = stepConfig[step.key] || {};
      const enabled = settings.enabled ?? false;

      const unitFor = (key) => {
        if (key === "min" || key === "max") {
          if (step.key === "speed_range") return { suffix: speedUnit, kind: "speed" };
          if (step.key === "moisture") return { suffix: "%" };
          return { suffix: rateUnit, kind: rateKind };
        }
        return PARAM_META[key] || { label: key };
      };

      const params = Object.keys(step.defaults).map((key) => {
        const raw = settings[key] !== undefined ? settings[key] : step.defaults[key];
        const meta = { ...(PARAM_META[key] || {}), ...unitFor(key) };
        const label = meta.label || (key === "min" ? "Minimum" : key === "max" ? "Maximum" : key);
        const id = `p-${step.key}-${key}`;

        if (meta.bool) {
          return `<label class="inline" style="margin-bottom:7px">
            <input type="checkbox" id="${id}"${raw ? " checked" : ""}>
            <span style="font-size:12px">${label}</span></label>`;
        }
        if (meta.choices) {
          return this.field(label, this.selectInput(id, meta.choices, raw));
        }
        const shown = meta.kind && raw != null
          ? Number(Units.convert[meta.kind](raw).toFixed(3)) : raw;
        return this.field(
          `${label}${meta.suffix ? ` <span style="color:var(--text-faint)">(${meta.suffix})</span>` : ""}`,
          `<input type="number" id="${id}" value="${shown ?? ""}" step="${meta.step || "any"}"
             data-kind="${meta.kind || ""}">`);
      }).join("");

      const node = document.createElement("details");
      node.className = "fold";
      node.innerHTML = `
        <summary>
          <label class="inline" style="display:inline-flex" onclick="event.stopPropagation()">
            <input type="checkbox" id="en-${step.key}"${enabled ? " checked" : ""}>
            <span>${this.escape(step.label)}</span>
          </label>
        </summary>
        <div class="inner">
          <p class="hint tight">${this.escape(step.description)}</p>
          ${params || '<p class="hint tight">No parameters.</p>'}
        </div>`;
      box.appendChild(node);
    }
  },

  collectCleanConfig() {
    const steps = {};
    for (const step of this.state.catalog.steps) {
      const entry = { enabled: this.checked(`en-${step.key}`) };
      for (const key of Object.keys(step.defaults)) {
        const node = document.getElementById(`p-${step.key}-${key}`);
        if (!node) continue;
        if (node.type === "checkbox") { entry[key] = node.checked; continue; }
        if (node.tagName === "SELECT") { entry[key] = node.value; continue; }
        if (node.value === "") { entry[key] = null; continue; }
        let value = Number(node.value);
        const kind = node.dataset.kind;
        if (kind && Units.toInternal[kind]) value = Units.toInternal[kind](value);
        entry[key] = value;
      }
      steps[step.key] = entry;
    }
    return {
      value_column: this.value("clean-column"),
      corrections: { flow_delay_s: this.number("clean-delay", 0) },
      steps,
    };
  },

  async runClean() {
    // A new cleaning replaces the clean dataset; the old comparison is stale.
    if (this.state.compare) this.exitCompare();
    const body = this.collectCleanConfig();
    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api(`/api/datasets/${this.state.selectedId}/clean`, { method: "POST", body }));
    if (!result) return;

    this.state.reports[`${this.state.selectedId}:clean`] = result;
    await this.refreshDatasets();
    this.renderCleanReport(result);
    this.toast("Cleaning done",
      `${Units.num(result.report.totals.removed, 0)} records removed ` +
      `(${Units.num(result.report.totals.removed_pct, 1)}%).`);
  },

  renderCleanReport(result) {
    const box = document.getElementById("clean-report");
    if (!box) return;
    const report = result.report;
    const totals = report.totals;
    const column = report.value_column;
    const { conv, unit } = Units.forColumn(column, this.state.selected?.meta?.operation);
    const before = report.statistics.before;
    const after = report.statistics.after;

    const stepRows = report.steps.filter((s) => !s.skipped).map((s) =>
      `<tr${s.removed ? ` class="clickable" data-reason="${this.escape(s.label)}"` : ""}>
       <td title="${this.escape(s.detail)}">${this.escape(s.label)}</td>
       <td class="num">${Units.num(s.removed, 0)}</td>
       <td class="num">${Units.num(s.removed / totals.input * 100, 1)}%</td>
       <td class="num">${Units.num(s.remaining, 0)}</td></tr>`).join("");

    const skipped = report.steps.filter((s) => s.skipped);

    const compare = (label, b, a, decimals, convert) => {
      const cb = convert ? convert(b) : b;
      const ca = convert ? convert(a) : a;
      let deltaText = "—";
      if (cb) {
        const delta = ((ca - cb) / Math.abs(cb)) * 100;
        const arrow = Math.abs(delta) < 0.05 ? "" : (delta > 0 ? "▲" : "▼");
        deltaText = `${arrow} ${Units.num(delta, 1)}%`;
      }
      return `<tr><td>${label}</td>
        <td class="num">${Units.num(cb, decimals)}</td>
        <td class="num">${Units.num(ca, decimals)}</td>
        <td class="num" style="color:var(--text-muted)">${deltaText}</td></tr>`;
    };

    box.innerHTML = `
      <div class="panel">
        ${this.printableHead("Cleaning report", "btn-print-clean")}
        ${report.findings.map((f) =>
          `<div class="note ${f.level}">${this.escape(f.text)}</div>`).join("")}
        <div class="stat-grid" style="margin-top:10px">
          <div class="stat"><div class="k">Came in</div>
            <div class="v">${Units.num(totals.input, 0)}</div><div class="d">records</div></div>
          <div class="stat"><div class="k">Kept</div>
            <div class="v">${Units.num(totals.kept, 0)}</div><div class="d">records</div></div>
          <div class="stat"><div class="k">Removed</div>
            <div class="v">${Units.num(totals.removed_pct, 1)}</div><div class="d">%</div></div>
          <div class="stat"><div class="k">Area worked</div>
            <div class="v">${Units.num(Units.convert.area(totals.area_ha_after))}</div>
            <div class="d">${Units.label.area()}</div></div>
        </div>
        ${report.corrections.length ? report.corrections.map((c) =>
          `<div class="note" style="margin-top:8px">${this.escape(c)}</div>`).join("") : ""}
      </div>

      <div class="panel">
        <h3>Before and after</h3>
        <button class="wide${this.state.compare ? "" : " primary"}" id="btn-compare">
          ${this.state.compare ? "Back to a single map" : "Compare before and after on the map"}
        </button>
        <p class="hint tight">Two maps side by side, on one colour scale, moving
        together. Hover a point to read its value.</p>
        <div class="scroll-x"><table class="data">
          <tr><th>Statistic</th><th>Before</th><th>After</th><th>Δ</th></tr>
          ${compare(`Mean (${unit})`, before.mean, after.mean, null, conv)}
          ${compare(`Median (${unit})`, before.median, after.median, null, conv)}
          ${compare(`SD (${unit})`, before.std, after.std, null, conv)}
          ${compare("Coeff. of variation (%)", before.cv, after.cv, 1)}
          ${compare(`Minimum (${unit})`, before.min, after.min, null, conv)}
          ${compare(`Maximum (${unit})`, before.max, after.max, null, conv)}
        </table></div>
        <h4>Distribution</h4>
        <div id="clean-hist"></div>
        <p class="hint tight">Grey: before. Green: after.</p>
      </div>

      <div class="panel">
        <h3>What each filter removed</h3>
        <div class="scroll-x"><table class="data">
          <tr><th>Filter</th><th>Removed</th><th>% of total</th><th>Left</th></tr>
          ${stepRows || '<tr><td colspan="4">No filter was enabled.</td></tr>'}
        </table></div>
        ${skipped.length ? `<details class="fold" style="margin-top:8px">
          <summary>${skipped.length} filter(s) could not run</summary>
          <div class="inner">${skipped.map((s) =>
            `<p class="hint tight"><b>${this.escape(s.label)}</b>: ${this.escape(s.detail)}</p>`
          ).join("")}</div></details>` : ""}
        <p class="hint tight">Click a filter to see on the map exactly which records
        it removed. Ten percent removed means nothing until you can see it was the
        headland and the overlap, and not a strip through the middle of the field.</p>
        <h4>Removals by reason</h4>
        <div id="clean-bars"></div>
        <div class="row tight" style="margin-top:8px">
          <button class="small" id="btn-show-all-removed">Show everything removed</button>
          <button class="small" id="btn-hide-removed">Clear the overlay</button>
        </div>
        <div id="removed-legend"></div>
      </div>

      <div class="panel">
        <h3>Datasets produced</h3>
        <div class="row tight">
          ${result.clean ? '<button id="btn-goto-clean">View clean data</button>' : ""}
          ${result.removed ? '<button id="btn-goto-removed">View removed</button>' : ""}
        </div>
      </div>`;

    Charts.histogram(
      document.getElementById("clean-hist"),
      this.scaleHistogram(report.histogram.before, conv),
      this.scaleHistogram(report.histogram.after, conv),
    );
    Charts.bars(document.getElementById("clean-bars"),
      report.by_reason.map((r) => ({ label: r.reason, value: r.records })));

    if (result.removed) {
      for (const row of box.querySelectorAll("tr.clickable")) {
        row.style.cursor = "pointer";
        row.addEventListener("click",
          () => this.showRemoved(result.removed.id, row.dataset.reason));
      }
      document.getElementById("btn-show-all-removed")?.addEventListener("click",
        () => this.showRemoved(result.removed.id, null));
      document.getElementById("btn-hide-removed")?.addEventListener("click",
        () => this.clearRemovedOverlay());
    }

    document.getElementById("btn-goto-clean")?.addEventListener("click",
      () => this.selectDataset(result.clean.id));
    document.getElementById("btn-goto-removed")?.addEventListener("click",
      () => this.selectDataset(result.removed.id));
    // The clean copy is the dataset that carries this report; the server keeps
    // a copy on the original too, which serves once the clean copy is gone.
    document.getElementById("btn-print-clean")?.addEventListener("click", (event) =>
      this.printReport(result.clean?.id || this.state.selectedId, event.currentTarget));
    document.getElementById("btn-compare")?.addEventListener("click", () => {
      if (this.state.compare) this.exitCompare();
      else this.enterCompare(result);
    });
  },

  /* Draw the removed records over the clean ones, coloured by the filter that
   * caught them. Seeing that the removals sit on the headland and the overlap
   * is the difference between trusting a cleaning and hoping. */
  async showRemoved(removedDatasetId, onlyReason) {
    // The overlay is drawn on the single map. Painted over the "Before" pane
    // it would hide the values the comparison exists to show, and the
    // tooltip there would read a removal reason where the legend reads yield.
    if (this.state.compare) this.exitCompare();
    const payload = await this.api(
      `/api/datasets/${removedDatasetId}/map?group_column=removal_reason`
    ).catch(() => null);
    if (!payload?.lon?.length) {
      this.toast("Nothing to show", "No removed record carries a position.", "warn");
      return;
    }

    let filtered = payload;
    if (onlyReason) {
      const keep = [];
      for (let i = 0; i < payload.groups.length; i++) {
        if (payload.groups[i] === onlyReason) keep.push(i);
      }
      filtered = {
        lon: keep.map((i) => payload.lon[i]),
        lat: keep.map((i) => payload.lat[i]),
        groups: keep.map(() => onlyReason),
      };
    }

    // One colour per filter, taken from the same ramp the value map uses so the
    // two never look like different applications.
    const reasons = [...new Set(filtered.groups.filter(Boolean))];
    const palette = {};
    reasons.forEach((reason, index) => {
      const c = MapView.rampColor(reasons.length === 1 ? 0.82
        : index / Math.max(1, reasons.length - 1));
      palette[reason] = `rgb(${c.join(",")})`;
    });

    MapView.setOverlay(filtered, palette);
    document.getElementById("removed-legend").innerHTML = `
      <p class="hint tight" style="margin-top:8px">
        ${Units.num(filtered.lon.length, 0)} removed record(s) on the map${
          onlyReason ? ` — ${this.escape(onlyReason)}` : ""}.</p>
      ${reasons.map((reason) => `
        <div class="inline" style="margin-bottom:3px">
          <span style="width:11px;height:11px;border-radius:50%;flex:none;
                background:${palette[reason]};border:1px solid rgba(255,255,255,.7)"></span>
          <span style="font-size:11px">${this.escape(reason)}</span>
        </div>`).join("")}`;
  },

  /* The overlay and the legend that explains its colours go together: one
   * without the other is either unexplained dots or a legend for nothing. */
  clearRemovedOverlay() {
    MapView.clearOverlay();
    const legend = document.getElementById("removed-legend");
    if (legend) legend.innerHTML = "";
  },

  /* The histogram arrives in internal units; the bin edges have to follow the
   * chosen unit, or the axis would read kg/ha on a chart labelled bu/ac. */
  scaleHistogram(hist, convert) {
    if (!hist?.edges?.length) return hist;
    return { counts: hist.counts, edges: hist.edges.map((e) => convert(e)) };
  },
});

/* ======================================================================
 * Before and after, side by side
 * ==================================================================== */

/* A second map pane beside the main one: the original on the left, the
 * clean copy on the right, locked together and on one colour scale. The
 * histogram says how much came out of the distribution; this says where. */
Object.assign(App, {
  async enterCompare(result) {
    if (this.state.compare) return;
    if (!result?.clean?.id) {
      this.toast("Nothing to compare", "Run the cleaning first.", "warn");
      return;
    }
    // A report can outlive the clean copy it describes — removed from the
    // session, or dropped by a reopen — and the server would answer 404 for
    // the "After" pane. Better to say so before building a pane for it.
    if (!this.state.datasets.some((d) => d.id === result.clean.id)) {
      this.toast("Nothing to compare",
        "The clean copy of this dataset is no longer loaded. Run the cleaning again.", "warn");
      return;
    }
    // The panes show the value on one scale; the removed-records overlay
    // would hide the "Before" values and contradict the legend.
    this.clearRemovedOverlay();

    const panes = document.getElementById("map-panes");
    const paneA = document.getElementById("map-pane");
    const paneB = document.createElement("div");
    paneB.className = "map-pane";
    paneB.id = "map-pane-b";
    paneB.innerHTML = `<div id="map-b" class="map"></div>
      <canvas id="point-canvas-b" class="point-canvas"></canvas>
      <div class="map-caption">After</div>`;
    panes.appendChild(paneB);
    panes.classList.add("compare");
    const captionA = document.createElement("div");
    captionA.className = "map-caption";
    captionA.textContent = "Before";
    paneA.appendChild(captionA);

    const view = createMapView("map-b", "point-canvas-b");
    view.init();
    view.setBasemap(this.checked("show-basemap"));
    // Both containers changed width: Leaflet has to re-measure before any
    // projection, or the right pane draws against the old, full-width size.
    MapView.invalidateSize();
    view.invalidateSize();
    const main = MapView.instance();
    view.instance().setView(main.getCenter(), main.getZoom(), { animate: false });
    const unlink = linkMapViews(MapView, view);

    this.state.compare = {
      originalId: this.state.selectedId,
      cleanId: result.clean.id,
      column: result.report.value_column,
      view, unlink,
    };
    // The panes are coloured by the cleaned variable, whatever the picker says.
    document.getElementById("color-column").disabled = true;
    this.syncCompareButton();
    await this.paintCompare();
  },

  /* The toggle is updated in place: re-rendering the tab would reset every
   * filter parameter the user has adjusted in the form above it. */
  syncCompareButton() {
    const button = document.getElementById("btn-compare");
    if (!button) return;
    const on = !!this.state.compare;
    button.textContent = on ? "Back to a single map" : "Compare before and after on the map";
    button.classList.toggle("primary", !on);
  },

  async paintCompare() {
    const c = this.state.compare;
    if (!c) return;
    const fetchMap = (id) =>
      this.api(`/api/datasets/${id}/map?column=${encodeURIComponent(c.column)}`);
    const [before, after] = await Promise.all([fetchMap(c.originalId), fetchMap(c.cleanId)])
      .catch((err) => { this.toast("That did not work", err.message, "error"); return []; });
    // The user may have left the comparison while the request was in flight.
    if (this.state.compare !== c) return;
    // Without both maps there is nothing to compare: an empty "After" pane
    // beside a disabled colour picker would look like a broken app rather
    // than a failed request, so the single map comes back.
    if (!before || !after) {
      this.exitCompare();
      return;
    }

    const { conv, unit } = Units.forColumn(c.column, this.state.selected?.meta?.operation);
    /* One scale for both, taken from the original: on separate scales the clean
     * map would simply look re-stretched, and the eye could not tell what was
     * removed from what was rescaled. */
    const scale = MapView.setPoints(before, conv, { unit });
    c.view.setPoints(after, conv, { unit, scale });
    this.renderLegend(c.column, unit, scale);
    document.getElementById("map-status").textContent =
      `Before: ${this.pointsStatus(before)} · After: ${this.pointsStatus(after)}`;
  },

  exitCompare({ reload = true } = {}) {
    const c = this.state.compare;
    if (!c) return;
    this.state.compare = null;
    c.unlink();
    c.view.destroy();
    document.getElementById("map-pane-b")?.remove();
    document.getElementById("map-panes").classList.remove("compare");
    document.querySelector("#map-pane > .map-caption")?.remove();
    document.getElementById("color-column").disabled = false;
    MapView.invalidateSize();
    this.syncCompareButton();
    if (reload) this.loadMap();
  },
});

/* ======================================================================
 * Economics
 *
 * On-farm trial analysis: the yield response to the rates the trial applied,
 * and the price ratio that says where the next unit of input stops paying
 * for itself. The tab key, the endpoint and the stored report keep the name
 * they were written with — see agrosuite/core/workflow.py.
 * ==================================================================== */

Object.assign(App, {
  tabDifm(panel) {
    const d = this.state.selected;
    if (!d) {
      const loaded = this.state.datasets.length;
      panel.innerHTML = loaded
        ? this.missingPanel(
            "Economics",
            "The analysis reads one table carrying both the applied rate and the "
            + "yield, and none is picked. Pick the trial file — or join the "
            + "as-applied and the yield map on the left, which builds that table.",
            { cta: "pick", label: "Show the first file" })
        : this.missingPanel(
            "Economics",
            "Nothing is open yet. This tab fits the yield response to the rates a "
            + "trial applied and finds the rate where the next unit of input stops "
            + "paying for itself. It needs a file carrying both the rate and the "
            + "yield — a cleaned trial, or the layers joined on a shared grid.",
            { cta: "load", label: "Load a file",
              hint: "A file that is already clean can come straight here: cleaning "
                    + "is not a step this tab waits for." });
      return;
    }

    const columns = d.numeric_columns || [];
    const guessRate = ["applied_rate", "target_rate", "rate"].find((c) => columns.includes(c))
      || columns[0];
    const guessValue = columns.includes("value") ? "value" : columns[0];
    const zoneOptions = [["", "— no zones —"],
      ...(d.columns || []).filter((c) => !["lon", "lat", "x", "y"].includes(c))
        .map((c) => [c, this.state.catalog.columns[c] || c])];

    const yieldUnit = Units.label.yield();
    const inputUnit = Units.label.inputRate();
    const yieldNum = Units.rateNumerator(yieldUnit);
    const inputNum = Units.rateNumerator(inputUnit);
    const symbol = Units.currencySymbol();
    const lengthUnit = Units.label.length();

    // Prices are set once on the project and reused here, so the same numbers
    // drive the panel on the left and the analysis on the right.
    const prices = this.state.project?.prices || {};
    const projectCrop = prices.crop_price
      ? Number(Units.priceFromInternal(prices.crop_price, yieldUnit).toFixed(2)) : 0;
    const projectInput = prices.input_cost
      ? Number(Units.priceFromInternal(prices.input_cost, inputUnit).toFixed(2)) : 0;

    // Without a crop price there is no economic optimum, only an agronomic
    // maximum, which is a different number. Saying so before the run beats
    // refusing after it.
    const needsPrices = !projectCrop;

    panel.innerHTML = `
      <div class="panel">
        <h3>Economic analysis: optimum rate from your trial</h3>
        <p class="hint tight">The yield response is fitted to the rates the trial
        applied; the crop price and the input cost turn that curve into the rate
        where the next unit stops paying for itself.</p>
      </div>

      ${needsPrices ? this.missingPanel(
        "The prices are missing",
        "Without a crop price the app can only report the agronomic maximum — the "
        + "rate that grows the most, not the rate that earns the most.",
        { cta: "prices", label: "Set the prices" }) : ""}

      <div class="panel">
        <h3>Trial</h3>
        ${this.field("Applied rate column",
          this.selectInput("difm-rate", this.columnOptions(), guessRate))}
        ${this.field("Yield column",
          this.selectInput("difm-value", this.columnOptions(), guessValue))}
        ${this.field("Zone column (optional)",
          this.selectInput("difm-zone", zoneOptions, ""),
          "With zones, the app compares a single rate against variable rate.")}
      </div>

      <div class="panel">
        <h3>Prices</h3>
        ${this.field(`Crop price (${symbol}/${yieldNum})`,
          this.numberInput("difm-price", projectCrop, "0.01", "0"))}
        ${this.field(`Input cost (${symbol}/${inputNum})`,
          this.numberInput("difm-cost", projectInput, "0.01", "0"),
          "The optimum rate is where the next unit of input stops paying for itself.")}
      </div>

      <div class="panel">
        <h3>Aggregation</h3>
        <div class="row tight">
          ${this.field(`Cell (${lengthUnit})`,
            this.numberInput("difm-cell", Math.round(Units.convert.length(20)), "1", "1"))}
          ${this.field(`Edge margin (${lengthUnit})`,
            this.numberInput("difm-edge", Math.round(Units.convert.length(6)), "1", "0"))}
        </div>
        <p class="hint tight">The margin drops the transition between strips, where the
        rates mix. It has to exceed half the pass spacing to have any effect.</p>
        ${this.field("Models to try", `<select id="difm-models" multiple size="4">${
          Object.entries(this.state.catalog.response_models).map(([k, v]) =>
            `<option value="${k}" selected>${this.escape(v)}</option>`).join("")}</select>`,
          "The one with the highest R-squared is chosen automatically.")}
      </div>

      <div class="panel">
        <button class="primary wide" id="btn-run-difm">Run the economic analysis</button>
      </div>

      <div id="difm-report"></div>

      ${this.producesPanel([
        { label: "Print the economic report (PDF)", cta: "print",
          hint: "The optimum rate, the curve it came from, the margin it earns and "
                + "the response of every rate the trial applied." },
        { label: "Build a prescription from the optimum", cta: "prescription",
          hint: "One rate for the field, or one per zone, carried to the Export tab "
                + "as the map to take to the monitor." },
      ], "The analysis is worth having on its own; the prescription is optional.")}`;

    document.getElementById("btn-run-difm").addEventListener("click", () => this.runDifm());
    // The cache key, the endpoint and the stored report key stay 'difm': they
    // are the wire names a saved project and a stored report were written
    // with. Only the label changed.
    const existing = this.state.reports[`${this.state.selectedId}:difm`];
    if (existing) this.renderDifmReport(existing);
    else if (d.has_difm_report) this.restoreDifmReport(d);
  },

  async runDifm() {
    const yieldUnit = Units.label.yield();
    const inputUnit = Units.label.inputRate();

    /* Prices come in per selling unit — dollars per bushel, per pound — and have
     * to become price per internal unit before going to the server. */
    const body = {
      rate_column: this.value("difm-rate"),
      value_column: this.value("difm-value"),
      zone_column: this.value("difm-zone") || null,
      crop_price: Units.priceToInternal(this.number("difm-price", 0), yieldUnit),
      input_cost: Units.priceToInternal(this.number("difm-cost", 0), inputUnit),
      cell_m: Units.toInternal.length(this.number("difm-cell", 20)),
      edge_margin_m: Units.toInternal.length(this.number("difm-edge", 6)),
      models: Array.from(document.getElementById("difm-models").selectedOptions)
        .map((o) => o.value),
    };
    if (!body.crop_price) {
      this.toast("Enter a price",
        "Without a crop price there is no economic optimum — only the agronomic "
        + "maximum, which is a different number.", "warn");
      return;
    }

    const report = await this.busy(document.getElementById("right-panel"), () =>
      this.api(`/api/datasets/${this.state.selectedId}/difm`, { method: "POST", body }));
    if (!report) return;
    this.state.reports[`${this.state.selectedId}:difm`] = report;
    this.renderDifmReport(report);
  },

  renderDifmReport(report) {
    const box = document.getElementById("difm-report");
    if (!box) return;

    const model = report.chosen_model;
    const economy = report.economics || {};
    const yieldUnit = Units.label.yield();
    const inputUnit = Units.label.inputRate();
    const symbol = Units.currencySymbol();
    const areaUnit = Units.label.area();

    /* The rate arrives in the internal mass-per-area unit; on screen it shows in
     * the input rate unit — lb/ac under the Canadian default. */
    const rate = (v) => Units.num(Units.convert.inputRate(v));
    const yld = (v) => Units.num(Units.convert.yield(v));
    /* Internal profit is per hectare; convert it to the chosen area unit. */
    const perArea = (v) => Units.money(v * Units.factor("area", Units.get().area_unit));

    const rateRows = (report.by_rate || []).map((r) =>
      `<tr><td class="num">${rate(r.rate)}</td>
       <td class="num">${Units.num(r.n, 0)}</td>
       <td class="num">${yld(r.mean_yield)}</td>
       <td class="num">${r.mean_profit != null ? perArea(r.mean_profit) : "—"}</td></tr>`).join("");

    const zones = report.zones;
    const zoneRows = (zones?.by_zone || []).map((z) => z.error
      ? `<tr><td>${this.escape(z.zone)}</td><td colspan="4">${this.escape(z.error)}</td></tr>`
      : `<tr><td>${this.escape(z.zone)}</td>
         <td class="num">${Units.num(z.cells, 0)}</td>
         <td class="num">${Units.num(z.r2, 3)}</td>
         <td class="num">${rate(z.optimum_rate)}</td>
         <td class="num">${perArea(z.profit_at_optimum)}</td></tr>`).join("");

    box.innerHTML = `
      <div class="panel">
        ${this.printableHead("Recommendation", "btn-print-difm")}
        <div class="stat-grid">
          <div class="stat"><div class="k">Economic optimum rate</div>
            <div class="v">${rate(economy.optimum_rate)}</div>
            <div class="d">${inputUnit}</div></div>
          <div class="stat"><div class="k">Expected yield</div>
            <div class="v">${yld(economy.yield_at_optimum)}</div>
            <div class="d">${yieldUnit}</div></div>
          <div class="stat"><div class="k">Margin at the optimum</div>
            <div class="v" style="font-size:13px">${perArea(economy.profit_at_optimum)}</div>
            <div class="d">per ${areaUnit}</div></div>
          <div class="stat"><div class="k">Agronomic maximum</div>
            <div class="v">${economy.agronomic_maximum != null
              ? rate(economy.agronomic_maximum) : "—"}</div>
            <div class="d">${inputUnit}</div></div>
        </div>
        ${economy.at_range_limit ? `<div class="note alert" style="margin-top:8px">
          The optimum landed at the edge of the tested range — the trial never showed
          the point of diminishing returns. Read it as "at least this much".</div>` : ""}
        ${(report.notes || []).map((o) =>
          `<div class="note" style="margin-top:8px">${this.escape(o)}</div>`).join("")}
        ${report.parameters ? `<p class="hint tight" style="margin-top:8px">
          ${Units.num(Units.convert.length(report.parameters.cell_m), 0)}
          ${Units.label.length()} cells · ${Units.num(
            Units.convert.length(report.parameters.edge_margin_m), 0)}
          ${Units.label.length()} edge margin.</p>` : ""}
      </div>

      <div class="panel">
        <h3>Response curve</h3>
        <div id="difm-curve"></div>
        <div class="stat-grid" style="margin-top:10px">
          <div class="stat"><div class="k">Model</div>
            <div class="v" style="font-size:13px">${this.escape(model.label)}</div>
            <div class="d">${model.n} cells</div></div>
          <div class="stat"><div class="k">R²</div>
            <div class="v">${Units.num(model.r2, 3)}</div>
            <div class="d">RMSE ${yld(model.rmse)} ${yieldUnit}</div></div>
        </div>
        ${model.message ? `<div class="note warning" style="margin-top:8px">${
          this.escape(model.message)}</div>` : ""}
        <details class="fold" style="margin-top:8px">
          <summary>All models tried</summary>
          <div class="inner scroll-x"><table class="data">
            <tr><th>Model</th><th>R²</th><th>RMSE</th></tr>
            ${(report.models_evaluated || []).map((m) =>
              `<tr><td>${this.escape(m.label)}</td>
               <td class="num">${Units.num(m.r2, 3)}</td>
               <td class="num">${yld(m.rmse)}</td></tr>`).join("")}
          </table></div>
        </details>
      </div>

      <div class="panel">
        <h3>Response by rate</h3>
        <div class="scroll-x"><table class="data">
          <tr><th>Rate (${inputUnit})</th><th>Cells</th>
              <th>Yield (${yieldUnit})</th><th>Margin/${areaUnit}</th></tr>
          ${rateRows}
        </table></div>
      </div>

      ${zones ? `<div class="panel">
        <h3>By zone · ${this.escape(zones.zone_column)}</h3>
        <div class="scroll-x"><table class="data">
          <tr><th>Zone</th><th>Cells</th><th>R²</th>
              <th>Optimum (${inputUnit})</th><th>Margin/${areaUnit}</th></tr>
          ${zoneRows}
        </table></div>
        ${zones.comparison?.reading ? `
          <div class="note ${zones.comparison.gain_per_ha > 0 ? "ok" : "warning"}"
               style="margin-top:10px">${this.escape(zones.comparison.reading)}</div>
          <div class="stat-grid" style="margin-top:8px">
            <div class="stat"><div class="k">Best single rate</div>
              <div class="v">${rate(zones.comparison.best_uniform_rate)}</div>
              <div class="d">${inputUnit}</div></div>
            <div class="stat"><div class="k">Variable rate gain</div>
              <div class="v" style="font-size:13px">${perArea(zones.comparison.gain_per_ha)}</div>
              <div class="d">per ${areaUnit}</div></div>
          </div>` : ""}
      </div>` : ""}

      <div class="panel">
        <button class="wide" id="btn-difm-to-rx">Build a prescription from this analysis</button>
      </div>`;

    Charts.responseCurve(document.getElementById("difm-curve"), report.curve, economy, {
      yieldConv: (v) => Units.convert.yield(v),
      rateConv: (v) => Units.convert.inputRate(v),
      yieldUnit,
      rateUnit: inputUnit,
    });

    document.getElementById("btn-difm-to-rx").addEventListener("click", () =>
      this.prescriptionFromDifm(report));
    document.getElementById("btn-print-difm").addEventListener("click", (event) =>
      this.printReport(this.state.selectedId, event.currentTarget));
  },

  /* Turn the analysis result into a prescription: the optimum per zone when
   * zones exist, or a single rate when they do not. */
  prescriptionFromDifm(report) {
    const zones = report.zones?.by_zone?.filter((z) => !z.error && z.optimum_rate != null);
    if (!zones?.length) {
      const rate = report.economics?.optimum_rate;
      if (rate == null) { this.toast("No rate", "The analysis produced no optimum rate.", "warn"); return; }
      this.state.pendingRx = { uniform: rate };
      this.toast("Rate carried to export",
        `Single rate of ${Units.num(Units.convert.inputRate(rate))} ${Units.label.inputRate()}. ` +
        "Go to the Export tab.");
    } else {
      this.state.pendingRx = {
        byZone: Object.fromEntries(zones.map((z) => [z.zone, z.optimum_rate])),
        zoneColumn: report.zones.zone_column,
      };
      this.toast("Zone rates carried to export",
        zones.map((z) => `${z.zone}: ${Units.num(Units.convert.inputRate(z.optimum_rate))}`).join(" · "));
    }
    this.goToTab("exportar");
  },
});

/* ======================================================================
 * Trial design
 *
 * The first half of the season's work, and it stands on its own: a boundary
 * in, the plots and the AB line out, straight to the monitor. Nothing here
 * asks for a yield map, for prices or for cleaning.
 * ==================================================================== */

Object.assign(App, {
  tabEnsaio(panel) {
    // The layout needs a field and nothing else. With no layer open and no
    // drawing on the map there is nowhere to put the strips, so the panel
    // asks for one instead of offering a form that cannot be submitted.
    if (!this.state.datasets.length && !this.state.drawing?.length) {
      panel.innerHTML = this.missingPanel(
        "Trial design",
        "There is no field to lay the strips on yet. A boundary file is the "
        + "truest outline, but any layer works — a yield map falls back to the "
        + "hull of its points — and you can also draw the field on the map.",
        { cta: "load", label: "Load a boundary or a field file",
          hint: "Nothing else is needed here: no yield map, no prices, no "
                + "cleaning. Rates, plot size, replications and direction are "
                + "set on this tab once there is a field." })
        + this.missingPanel(
          "Or draw it",
          "Click the field's corners on the map and the layout is built inside "
          + "what you drew.",
          { cta: "draw", label: "Draw the field on the map" });
      return;
    }

    const lengthUnit = Units.label.length();
    const inputUnit = Units.label.inputRate();
    const areaUnit = Units.label.area();
    const datasets = this.state.datasets.map((d) => [d.id, d.label]);
    /* The usual implement width in the chosen unit: 60 ft is the standard
     * drill and boom on the Prairies; 12 m, the common metric equivalent. */
    const defaultWidth = Units.label.length() === "ft" ? 60
      : Math.round(Units.convert.length(12));

    /* Suggested rates around a typical one, in the user's unit — a useful trial
     * has to reach well below and well above current practice. The step is
     * rounded to a round number in the chosen unit: nobody programs a monitor
     * at 44.8 lb/ac. */
    const typical = Units.convert.inputRate(100);
    const step = this.roundStep(typical / 2);
    const suggested = [0, 1, 2, 3, 4].map((i) => i * step).join(", ");

    panel.innerHTML = `
      <div class="panel">
        <h3>Field boundary</h3>
        ${this.field("From a loaded dataset", this.selectInput("design-source",
          [["", "— choose —"], ...datasets], this.state.selectedId || ""),
          "Use a boundary shapefile, or a yield map (the app falls back to the hull of the points).")}
        <div class="row tight">
          <button class="small" id="btn-draw-boundary">Draw on the map</button>
          <button class="small" id="btn-clear-boundary">Clear drawing</button>
        </div>
        <p class="hint tight" id="draw-status"></p>
      </div>

      <div class="panel">
        <h3>Treatments</h3>
        ${this.field(`Rates to test (${inputUnit}, comma separated)`,
          `<input type="text" id="design-rates" value="${suggested}">`,
          "At least three, and it is worth including an unapplied strip as a check.")}
        ${this.machinePicker("design")}
        <div class="row tight">
          ${this.field(`Implement width (${lengthUnit})`,
            this.numberInput("design-width", defaultWidth, "1", "1"))}
          ${this.field("Passes per strip",
            this.numberInput("design-passes", 2, "1", "1"))}
        </div>
        <div class="row tight">
          ${this.field("Blocks (replicates)", this.numberInput("design-blocks", 4, "1", "1"))}
          ${this.field(`Headland setback (${lengthUnit})`,
            this.numberInput("design-buffer", Math.round(Units.convert.length(15)), "1", "0"))}
        </div>
        ${this.field("Strip direction (degrees, empty = longest side)",
          `<input type="number" id="design-angle" step="1" placeholder="automatic">`)}
      </div>

      <div class="panel">
        <button class="primary wide" id="btn-run-design">Generate layout</button>
        <p class="hint tight">Two passes per strip leave the centre pass free of the
        neighbouring rates — that is what lets the analysis drop the edges without
        losing the whole treatment.</p>
      </div>

      <div id="design-report"></div>

      ${this.producesPanel([
        { label: "Build the package for the monitor", cta: "export-package",
          hint: "The USB folder the terminal reads: the strips as the prescription, "
                + "the boundary and the AB line, checked before the stick leaves." },
      ], "Planning the trial is a season's work on its own — it ends here, at the "
         + "monitor, months before there is anything to analyse.")}`;

    this.bindMachinePicker("design");
    document.getElementById("btn-run-design").addEventListener("click", () => this.runDesign());
    document.getElementById("btn-draw-boundary").addEventListener("click", () => this.startDrawing());
    document.getElementById("btn-clear-boundary").addEventListener("click", () => {
      this.state.drawing = null;
      MapView.clearOverlays();
      document.getElementById("draw-status").textContent = "";
    });

    if (this.state.design) {
      this.renderDesignReport(this.state.design);
      // The strips go back on the map with the report: after a reopened
      // project, or a visit to a tab that painted its own overlays, the
      // report alone would describe a layout nobody can see. A drawing in
      // progress is left alone — it is the boundary of the next layout.
      if (!this.state.drawing) this.drawDesignWithGuidance();
    }
  },

  startDrawing() {
    this.state.drawing = [];
    const status = document.getElementById("draw-status");
    status.textContent = "Click the field's corners, then press Generate layout.";

    const handler = (event) => {
      if (!this.state.drawing) { MapView.offClick(handler); return; }
      this.state.drawing.push([event.latlng.lng, event.latlng.lat]);
      MapView.setPolygons([this.state.drawing.concat([this.state.drawing[0]])], { fill: true });
      status.textContent =
        `${this.state.drawing.length} corner(s). At least 3 are needed to close the field.`;
    };
    MapView.onClick(handler);
  },

  async runDesign() {
    const inputUnit = Units.label.inputRate();
    const rates = (this.value("design-rates") || "")
      .split(/[,;]/).map((s) => Number(s.trim().replace(",", ".")))
      .filter((n) => Number.isFinite(n))
      .map((n) => Units.toInternal.inputRate(n));

    if (rates.length < 3) {
      this.toast("Not enough rates",
        `Enter at least three rates in ${inputUnit}.`, "warn");
      return;
    }

    const body = {
      rates,
      implement_width_m: Units.toInternal.length(this.number("design-width", 12)),
      passes_per_strip: this.number("design-passes", 2),
      blocks: this.number("design-blocks", 4),
      buffer_m: Units.toInternal.length(this.number("design-buffer", 0)),
      angle_deg: this.number("design-angle", null),
    };
    if (this.state.drawing?.length >= 3) body.boundary = this.state.drawing;
    else if (this.value("design-source")) body.boundary_dataset_id = this.value("design-source");
    else { this.toast("No boundary", "Pick a dataset or draw the field.", "warn"); return; }

    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api("/api/design", { method: "POST", body }));
    if (!result) return;

    this.state.design = result;
    MapView.setFeatures(result.features, "rate");
    MapView.fitOverlays();
    this.renderDesignReport(result);
    this.toast("Trial laid out",
      `${result.summary.strips} strips in ${result.summary.blocks} block(s).`);
  },

  renderDesignReport(result) {
    const box = document.getElementById("design-report");
    if (!box) return;
    const s = result.summary;
    const inputUnit = Units.label.inputRate();
    const lengthUnit = Units.label.length();
    const areaUnit = Units.label.area();

    const reps = Object.entries(s.reps_per_rate).map(([rateValue, count]) =>
      `<tr><td class="num">${Units.num(Units.convert.inputRate(Number(rateValue)))}</td>
       <td class="num">${count}</td></tr>`).join("");

    box.innerHTML = `
      <div class="panel">
        <h3>Layout generated</h3>
        ${(result.warnings || []).map((w) =>
          `<div class="note warning">${this.escape(w)}</div>`).join("")}
        <div class="stat-grid" style="margin-top:8px">
          <div class="stat"><div class="k">Strips</div>
            <div class="v">${s.strips}</div><div class="d">${s.blocks} block(s)</div></div>
          <div class="stat"><div class="k">Strip width</div>
            <div class="v">${Units.num(Units.convert.length(s.strip_width_m), 0)}</div>
            <div class="d">${lengthUnit} · ${s.passes_per_strip} pass(es)</div></div>
          <div class="stat"><div class="k">Trial area</div>
            <div class="v">${Units.num(Units.convert.area(s.total_area_ha))}</div>
            <div class="d">${areaUnit}</div></div>
          <div class="stat"><div class="k">Direction</div>
            <div class="v">${Units.num(s.direction_deg, 0)}°</div>
            <div class="d">azimuth</div></div>
        </div>
        <h4>Replicates per rate</h4>
        <table class="data">
          <tr><th>Rate (${inputUnit})</th><th>Strips</th></tr>${reps}
        </table>
      </div>
      <div class="panel">
        <h3>Guidance line</h3>
        <p class="hint tight">The AB line has to follow the same direction as the
        strips; if the operator enters at another angle, the passes cut across the
        treatments and the trial is lost.</p>
        <button class="wide" id="btn-make-ab" style="margin-bottom:6px">
          Build an AB line along the trial direction
        </button>
        <div id="ab-status"></div>
      </div>
      <div class="panel">
        <button class="primary wide" id="btn-design-export">Export this trial</button>
      </div>`;

    document.getElementById("btn-make-ab").addEventListener("click", () => this.makeGuidance());
    if (this.state.guidance?.length) this.renderGuidanceStatus();

    document.getElementById("btn-design-export").addEventListener("click", () => {
      this.state.exportMode = "package";
      this.goToTab("exportar");
    });
  },
});

Object.assign(App, {
  async makeGuidance() {
    if (!this.state.design) {
      this.toast("No trial yet", "Generate the layout before building the AB line.", "warn");
      return;
    }
    const body = {
      angle_deg: this.state.design.summary.direction_deg,
      name: "Trial AB line",
    };
    if (this.state.drawing?.length >= 3) body.boundary = this.state.drawing;
    else body.boundary_dataset_id = this.value("design-source") || this.state.selectedId;

    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api("/api/guidance", { method: "POST", body }));
    if (!result) return;

    this.state.guidance = [result.line];
    this.renderGuidanceStatus();
    this.drawDesignWithGuidance();
    this.toast("AB line created", `Compass bearing ${Units.num(result.line.heading, 1)}°.`);
  },

  renderGuidanceStatus() {
    const box = document.getElementById("ab-status");
    if (!box) return;
    box.innerHTML = (this.state.guidance || []).map((line) => `
      <div class="note ok">
        <b>${this.escape(line.name)}</b> — bearing ${Units.num(line.heading, 1)}°<br>
        <span style="font:11px var(--mono)">
          A ${line.a[1].toFixed(6)}, ${line.a[0].toFixed(6)}<br>
          B ${line.b[1].toFixed(6)}, ${line.b[0].toFixed(6)}
        </span>
      </div>`).join("");
  },

  /* Draw the strips and the AB line on top — the visual check that the two are
   * pointing the same way. */
  drawDesignWithGuidance() {
    if (!this.state.design) return;
    MapView.setFeatures(this.state.design.features, "rate");
    for (const line of this.state.guidance || []) {
      MapView.addLine([line.a, line.b], line.name);
    }
  },
});

/* ======================================================================
 * Export
 * ==================================================================== */

Object.assign(App, {
  tabExportar(panel) {
    const hasDesign = !!this.state.design;
    const hasDataset = !!this.state.selectedId;
    const mode = this.state.exportMode || "package";
    const inputUnit = Units.label.inputRate();

    // Nothing to write out yet: the tab says which of the two things it
    // takes is missing, rather than drawing a form over an empty session.
    if (!hasDesign && !hasDataset) {
      panel.innerHTML = this.missingPanel(
        "Export",
        this.state.datasets.length
          ? "There is a trial layout or a file to write, but none is picked."
          : "There is nothing to write out yet. This tab builds the package the "
            + "monitor reads — boundary, AB lines and prescription in the layout "
            + "the platform expects — or individual files from any layer that is "
            + "open.",
        this.state.datasets.length
          ? { cta: "pick", label: "Show the first file" }
          : { cta: "load", label: "Load a file",
              hint: "A trial laid out on the Trial design tab is exported from "
                    + "here too, with no data loaded at all." });
      return;
    }

    panel.innerHTML = `
      <div class="panel">
        <h3>How to export</h3>
        ${this.field("Mode", this.selectInput("exp-mode", [
          ["package", "Ready-to-load package for the monitor"],
          ["files", "Individual files"],
        ], mode), mode === "package"
          ? "Builds the USB folder in the layout the monitor looks for, with boundary, AB lines and prescription."
          : "Generates only the chosen files, with no folder structure.")}
      </div>
      <div id="export-form"></div>
      <div id="export-report"></div>`;

    document.getElementById("exp-mode").addEventListener("change", (event) => {
      this.state.exportMode = event.target.value;
      this.renderTab();
    });

    if (mode === "package") this.renderPackageForm();
    else this.renderFilesForm();
  },

  /* ---------------------------------------------------- package mode --- */

  renderPackageForm() {
    const box = document.getElementById("export-form");
    const monitors = this.state.catalog.monitors;
    const chosen = this.state.exportMonitor || "john_deere";
    const lengthUnit = Units.label.length();
    const hasDesign = !!this.state.design;
    const hasGuidance = !!this.state.guidance?.length;
    const datasets = this.state.datasets.map((d) => [d.id, d.label]);

    box.innerHTML = `
      <div class="panel">
        <h3>Target monitor</h3>
        ${this.field("Platform", this.selectInput("pkg-monitor",
          monitors.map((m) => [m.key, m.label]), chosen))}
        ${this.machinePicker("package")}
        <div id="pkg-accepts"></div>
      </div>

      <div class="panel">
        <h3>What goes in the package</h3>
        <label class="inline" style="margin-bottom:7px">
          <input type="checkbox" id="pkg-rx"${hasDesign ? " checked" : ""}
            ${hasDesign ? "" : "disabled"}>
          <span>Prescription${hasDesign
            ? ` — ${this.state.design.summary.strips} trial strips`
            : " (lay the trial out first)"}</span></label>
        <label class="inline" style="margin-bottom:7px">
          <input type="checkbox" id="pkg-boundary" checked>
          <span>Field boundary</span></label>
        <label class="inline" style="margin-bottom:7px">
          <input type="checkbox" id="pkg-guidance"${hasGuidance ? " checked" : ""}
            ${hasGuidance ? "" : "disabled"}>
          <span>AB lines${hasGuidance
            ? ` — ${this.state.guidance.length} line(s)`
            : " (build them on the Trial design tab)"}</span></label>
        ${this.field("Boundary from", this.selectInput("pkg-boundary-src",
          [["", "— current drawing / trial —"], ...datasets],
          this.state.selectedId || ""),
          "A boundary imported from the monitor is truer than the hull of the points.")}
      </div>

      <div class="panel">
        <h3>Rate</h3>
        ${this.field("Kind", this.selectInput("pkg-kind",
          Object.entries(this.state.catalog.rate_kinds), "mass"))}
        ${this.field("Unit written to the file", this.selectInput("pkg-unit",
          this.state.units.groups.rate_mass.units.map((u) => [u.key, u.label]),
          Units.label.inputRate()))}
        ${this.field("Crop (for bushel units)", this.selectInput("pkg-crop",
          this.state.units.crops.map((c) => [c.key, c.label]), Units.get().crop))}
        ${this.field(`ISOXML grid cell (${lengthUnit})`,
          this.numberInput("pkg-cell", Math.round(Units.convert.length(10)), "1", "1"),
          "A smaller cell makes a bigger file; 10 m suits nearly every terminal.")}
      </div>

      <div class="panel">
        <h3>Naming</h3>
        <div class="row tight">
          ${this.field("Field", `<input type="text" id="pkg-field" value="Field">`)}
          ${this.field("Task", `<input type="text" id="pkg-task" value="Prescription">`)}
        </div>
        <div class="row tight">
          ${this.field("Client", `<input type="text" id="pkg-customer" value="AgroSuite">`)}
          ${this.field("Farm", `<input type="text" id="pkg-farm" value="Farm">`)}
        </div>
        ${this.field("Product", `<input type="text" id="pkg-product" value="Product">`)}
      </div>

      <div class="panel">
        <button class="primary wide" id="btn-run-package">Build the package</button>
      </div>

      ${this.qgisPanel()}`;

    const showAccepts = () => {
      const monitor = monitors.find((m) => m.key === this.value("pkg-monitor"));
      const node = document.getElementById("pkg-accepts");
      if (!monitor) { node.innerHTML = ""; return; }
      const labels = this.state.catalog.artifact_labels;
      node.innerHTML = `
        <p class="hint tight">Accepts: ${monitor.accepts.map((a) => labels[a] || a).join(" · ")}</p>
        <p class="hint tight">Rate field in the shapefile: <b>${monitor.rate_field}</b></p>
        <details class="fold" style="margin-top:6px">
          <summary>How to load it on the ${this.escape(monitor.label)}</summary>
          <div class="inner">${monitor.instructions.map((i) =>
            `<p class="hint tight">• ${this.escape(i)}</p>`).join("")}</div>
        </details>`;
      /* Disable what the platform cannot take, rather than letting the user tick
       * it and only find out afterwards that nothing was generated. */
      for (const [id, artifact] of [["pkg-rx", "prescription"],
                                    ["pkg-boundary", "boundary"],
                                    ["pkg-guidance", "guidance"]]) {
        const input = document.getElementById(id);
        const supported = monitor.accepts.includes(artifact);
        if (!supported) { input.checked = false; input.disabled = true; }
        input.parentElement.style.opacity = supported ? "1" : "0.45";
      }
    };

    document.getElementById("pkg-monitor").addEventListener("change", (event) => {
      this.state.exportMonitor = event.target.value;
      showAccepts();
    });
    document.getElementById("pkg-kind").addEventListener("change", () => {
      const kind = this.value("pkg-kind");
      const group = kind === "volume" ? "rate_volume" : kind === "count" ? "rate_count" : "rate_mass";
      const select = document.getElementById("pkg-unit");
      select.innerHTML = this.state.units.groups[group].units.map((u) =>
        `<option value="${u.key}">${this.escape(u.label)}</option>`).join("");
    });
    document.getElementById("btn-run-package").addEventListener("click", () => this.runPackage());
    this.bindQgisPanel();
    this.bindMachinePicker("package");
    showAccepts();
  },

  /* QGIS is where the layers get looked at properly — against imagery, against
   * soil maps, with zones drawn by hand. The GeoPackage is the reliable half of
   * the bridge; the project file just saves adding each layer by hand. */
  qgisPanel() {
    return `
      <div class="panel">
        <h3>QGIS</h3>
        <div class="row tight">
          <button id="btn-qgis-export">Write layers for QGIS</button>
          <button id="btn-qgis-import">Open a QGIS project</button>
        </div>
        <p class="hint tight">Export writes one GeoPackage with every loaded layer,
        plus a .qgs project that opens them all at once. Import reads a .qgs or .qgz
        and brings its layers in — which is how zones drawn in QGIS reach the monitor.</p>
        <div id="qgis-result"></div>
      </div>`;
  },

  bindQgisPanel() {
    document.getElementById("btn-qgis-export")?.addEventListener("click", async () => {
      const result = await this.busy(document.getElementById("qgis-result"), () =>
        this.api("/api/qgis/export", {
          method: "POST",
          body: { name: this.state.project?.name || "agrosuite" },
        }));
      if (!result) return;
      document.getElementById("qgis-result").innerHTML = `
        <div class="note ok" style="margin-top:8px">
          <b>${result.geopackage.layers.length} layer(s) written.</b><br>
          <span style="font:11px var(--mono)">${this.escape(result.folder)}</span>
        </div>
        <div class="scroll-x" style="margin-top:8px"><table class="data">
          <tr><th>Layer</th><th>Features</th><th>Type</th></tr>
          ${result.geopackage.layers.map((l) => `<tr>
            <td>${this.escape(l.layer)}</td>
            <td class="num">${Units.num(l.features, 0)}</td>
            <td>${this.escape(l.geometry)}</td></tr>`).join("")}
        </table></div>
        <p class="hint tight">${this.escape(result.note)}</p>
        <a class="btn wide" style="display:block;text-decoration:none;margin-top:8px"
           href="${result.download_url}" download>Download the GeoPackage</a>`;
      this.toast("Written for QGIS", `${result.geopackage.layers.length} layer(s).`);
    });

    document.getElementById("btn-qgis-import")?.addEventListener("click", () => {
      this.state.qgisImport = true;
      document.getElementById("dlg-path").showModal();
      document.getElementById("path-input").value = "";
      this.browse("");
      this.toast("Pick a project", "Choose a .qgs or .qgz file, then press Open.");
    });
  },

  async importQgisProject(path) {
    const project = await this.api(
      `/api/qgis/project?path=${encodeURIComponent(path)}`
    ).catch((err) => { this.toast("That did not work", err.message, "error"); return null; });
    if (!project) return;

    const importable = project.layers.filter((l) => l.importable);
    if (!importable.length) {
      this.toast("Nothing to import",
        "None of the project's layers are file-based layers on this machine.", "warn");
      return;
    }
    const result = await this.busy(document.querySelector("aside.left"), () =>
      this.api("/api/qgis/import", { method: "POST", body: { path } }));
    if (!result) return;

    await this.refreshDatasets();
    await this.selectDataset(result.imported[0].id);
    this.toast(
      `Imported from ${result.project}`,
      result.imported.map((d) => d.label).join(", ") +
      (result.skipped.length ? `\nSkipped: ${result.skipped.map((s) =>
        `${s.name} (${s.reason})`).join(", ")}` : ""),
    );
  },

  async runPackage() {
    const body = {
      monitor: this.value("pkg-monitor"),
      rate_property: "rate",
      rate_kind: this.value("pkg-kind"),
      rate_unit: this.value("pkg-unit"),
      crop: this.value("pkg-crop"),
      cell_m: Units.toInternal.length(this.number("pkg-cell", 10)),
      field_name: this.value("pkg-field") || "Field",
      task_name: this.value("pkg-task") || "Prescription",
      product_name: this.value("pkg-product") || "Product",
      customer_name: this.value("pkg-customer") || "AgroSuite",
      farm_name: this.value("pkg-farm") || "Farm",
    };

    if (this.checked("pkg-rx") && this.state.design) {
      body.features = this.state.design.features;
    }
    if (this.checked("pkg-guidance") && this.state.guidance?.length) {
      body.guidance_lines = this.state.guidance;
    }
    if (this.checked("pkg-boundary")) {
      const source = this.value("pkg-boundary-src");
      if (this.state.drawing?.length >= 3) body.boundary = this.state.drawing;
      else if (source) body.boundary_dataset_id = source;
      else if (this.state.design) body.features = body.features || this.state.design.features;
    }
    if (!body.features && !body.boundary && !body.boundary_dataset_id &&
        !body.guidance_lines?.length) {
      this.toast("Empty package", "Tick at least one item to include.", "warn");
      return;
    }

    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api("/api/export/package", { method: "POST", body }));
    if (!result) return;
    this.renderPackageReport(result);
    this.bindUsbPanel(result.folder);
    this.toast("Package built", `${result.contents.length} item(s) for the ${result.monitor_label}.`);
  },

  renderPackageReport(result) {
    const box = document.getElementById("export-report");
    const labels = this.state.catalog.artifact_labels;

    box.innerHTML = `
      <div class="panel">
        <h3>Package for the ${this.escape(result.monitor_label)}</h3>
        <a class="btn primary wide" style="display:block;text-decoration:none"
           href="${result.bundle.download_url}" download>
          Download ${this.escape(result.bundle.filename)}
        </a>
        <p class="hint tight">Written to:<br>
          <code style="font-size:11px">${this.escape(result.folder)}</code></p>
        <div class="scroll-x" style="margin-top:10px"><table class="data">
          <tr><th>File</th><th>Contents</th></tr>
          ${result.contents.map((c) => `<tr>
            <td style="font-family:var(--mono);font-size:11px">${this.escape(c.path)}</td>
            <td style="text-align:left">${this.escape(
              c.artifact === "isoxml" ? "ISOXML" : (labels[c.artifact] || c.artifact))}
              — ${this.escape(c.detail)}</td></tr>`).join("")}
        </table></div>
        ${(result.notes || []).map((o) =>
          `<div class="note" style="margin-top:8px">${this.escape(o)}</div>`).join("")}
        ${(result.skipped || []).map((s) =>
          `<div class="note warning" style="margin-top:6px">${this.escape(s)}</div>`).join("")}
      </div>
      ${this.verificationPanel(result.verification)}
      ${this.usbPanel(result)}
      <div class="panel">
        <h3>How to load it</h3>
        ${result.instructions.map((step, i) =>
          `<p class="hint tight">${i + 1}. ${this.escape(step)}</p>`).join("")}
        <p class="hint tight" style="margin-top:8px">The <b>README.txt</b> inside the
        package repeats these instructions, to read in the cab.</p>
      </div>`;
  },

  /* Copying to the stick is the last step of every job, and the step where
   * the mistakes happen — the folder ends up one level deep, or last season's
   * prescription is still sitting there. */
  usbPanel(result) {
    return `
      <div class="panel">
        <h3>Write to a USB drive</h3>
        <div class="row tight">
          <select id="usb-drive"><option value="">— refresh to list drives —</option></select>
          <button class="small" id="btn-usb-refresh" style="flex:none">Refresh</button>
        </div>
        <div id="usb-detail"></div>
        <button class="wide" id="btn-usb-write" style="margin-top:8px" disabled>
          Copy the package to the drive
        </button>
        <p class="hint tight">The package goes to the root of the drive, which is where
        the terminal looks. Anything already there is left alone unless you say
        otherwise.</p>
      </div>`;
  },

  bindUsbPanel(folder) {
    const select = document.getElementById("usb-drive");
    const detail = document.getElementById("usb-detail");
    const writeButton = document.getElementById("btn-usb-write");
    if (!select) return;

    const refresh = async () => {
      const payload = await this.api("/api/usb").catch(() => null);
      const drives = payload?.drives || [];
      select.innerHTML = drives.length
        ? drives.map((d) => `<option value="${this.escape(d.path)}">${
            this.escape(d.label)} — ${this.escape(d.path)}${
            d.removable ? "" : " (fixed disk)"}</option>`).join("")
        : '<option value="">no drive found</option>';
      select.dispatchEvent(new Event("change"));
    };

    const preview = async () => {
      const drive = select.value;
      if (!drive) { detail.innerHTML = ""; writeButton.disabled = true; return; }
      const plan = await this.api("/api/usb/plan", {
        method: "POST", body: { folder, drive },
      }).catch(() => null);
      if (!plan) { detail.innerHTML = ""; writeButton.disabled = true; return; }

      this.state.usbPlan = plan;
      writeButton.disabled = !plan.fits;
      detail.innerHTML = `
        <p class="hint tight">${plan.total_mb} MB to copy${
          plan.free_mb != null ? `, ${Units.num(plan.free_mb, 0)} MB free` : ""}.</p>
        ${plan.conflicts.length ? `
          <div class="note warning">
            Already on this drive: ${plan.conflicts.map((c) =>
              `<b>${this.escape(c.name)}</b>`).join(", ")}.
            <label class="inline" style="margin-top:6px">
              <input type="checkbox" id="usb-replace">
              <span>Replace ${plan.conflicts.length === 1 ? "it" : "them"}</span>
            </label>
          </div>` : ""}
        ${!plan.fits ? '<div class="note alert">Not enough room on the drive.</div>' : ""}`;
    };

    document.getElementById("btn-usb-refresh").addEventListener("click", refresh);
    select.addEventListener("change", preview);
    writeButton.addEventListener("click", async () => {
      const plan = this.state.usbPlan;
      const replaceAll = this.checked("usb-replace");
      if (plan?.conflicts.length && !replaceAll) {
        this.toast("Already on the drive",
          "Tick the replace box, or the existing folders stay as they are.", "warn");
      }
      const result = await this.busy(document.getElementById("export-report"), () =>
        this.api("/api/usb/write", {
          method: "POST",
          body: {
            folder, drive: select.value,
            replace: replaceAll ? plan.conflicts.map((c) => c.name) : [],
          },
        }));
      if (!result) return;
      detail.innerHTML += `<div class="note ok" style="margin-top:8px">${
        this.escape(result.message)}${result.skipped.length
          ? `<br>Left alone: ${result.skipped.map((s) =>
              this.escape(s.name)).join(", ")}.` : ""}</div>`;
      this.toast("Copied to the drive", result.message);
      this.refreshProject();
    });

    refresh();
  },

  /* Verification runs over the files already written. Red is a real blocker;
   * amber depends on the display's firmware. */
  verificationPanel(check) {
    if (!check) return "";
    const noteClass = { ok: "ok", warning: "warning", fail: "alert" }[check.verdict] || "";
    const icon = { ok: "✓", warning: "!", fail: "✕" };

    const groups = (check.groups || []).map((group) => {
      const failed = group.checks.filter((c) => c.status !== "ok");
      const rows = group.checks.map((c) => `
        <div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid var(--border)">
          <span style="flex:none;width:14px;text-align:center;font-weight:700;color:${
            c.status === "ok" ? "var(--ok)"
            : c.status === "warning" ? "var(--warn)" : "var(--danger)"
          }">${icon[c.status]}</span>
          <span style="min-width:0">
            <b style="font-size:12px">${this.escape(c.item)}</b>
            <span style="font-size:12px;color:var(--text-muted)"> — ${this.escape(c.message)}</span>
            ${c.fix ? `<br><span class="hint tight">${this.escape(c.fix)}</span>` : ""}
          </span>
        </div>`).join("");
      return `<details class="fold"${failed.length ? " open" : ""}>
        <summary>${this.escape(group.file)} · ${this.escape(group.kind)}${
          failed.length ? ` — ${failed.length} point(s) to look at` : " — all checked"}</summary>
        <div class="inner">${rows}</div>
      </details>`;
    }).join("");

    return `
      <div class="panel">
        <h3>Check before you take it out</h3>
        <div class="note ${noteClass}">${this.escape(check.summary)}</div>
        <div class="stat-grid" style="margin:10px 0">
          <div class="stat"><div class="k">Checked</div>
            <div class="v" style="color:var(--ok)">${check.totals.ok}</div>
            <div class="d">no reservations</div></div>
          <div class="stat"><div class="k">To confirm</div>
            <div class="v" style="color:var(--warn)">${check.totals.warning}</div>
            <div class="d">on the monitor screen</div></div>
          <div class="stat"><div class="k">Blockers</div>
            <div class="v" style="color:var(--danger)">${check.totals.fail}</div>
            <div class="d">the monitor would refuse</div></div>
          <div class="stat"><div class="k">Target</div>
            <div class="v" style="font-size:12px">${this.escape(check.monitor_label)}</div>
            <div class="d">profile applied</div></div>
        </div>
        ${groups}
        <p class="hint tight" style="margin-top:10px">${this.escape(check.caveat)}</p>
      </div>`;
  },

  /* -------------------------------------------- individual files mode -- */

  renderFilesForm() {
    const box = document.getElementById("export-form");
    const hasDesign = !!this.state.design;
    const hasDataset = !!this.state.selectedId;
    const brands = this.state.catalog.brands.filter((b) => b.export_formats.length);
    const source = this.state.exportSource || (hasDesign ? "design" : "dataset");
    const lengthUnit = Units.label.length();
    /* The name becomes the file name. A trial layout is a prescription; a
     * dataset exported as it stands is that dataset, and calling the zip
     * "Prescription" would misname what is inside it. */
    const defaultName = source === "dataset" && this.state.selected
      ? (this.state.selected.label.replace(/[^\w .\-]+/g, " ").replace(/\s+/g, " ")
          .trim().slice(0, 40) || "Data")
      : "Prescription";

    box.innerHTML = `
      <div class="panel">
        <h3>Source</h3>
        ${this.field("What to export", this.selectInput("exp-source", [
          ...(hasDesign ? [["design", "Trial design (prescription)"]] : []),
          ...(hasDataset ? [["dataset", `Data: ${this.state.selected?.label ?? ""}`]] : []),
        ], source))}
      </div>
      <div class="panel">
        <h3>Formats</h3>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="fmt-shapefile" checked><span>Shapefile (.shp)</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="fmt-isoxml"><span>ISOXML / TASKDATA</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="fmt-csv"><span>CSV</span></label>
        <label class="inline" style="margin-bottom:6px">
          <input type="checkbox" id="fmt-geojson"><span>GeoJSON</span></label>
      </div>
      <div class="panel">
        <h3>Settings</h3>
        ${this.field("Name the rate field for", this.selectInput("exp-brand",
          brands.map((b) => [b.key, b.label]), "john_deere"))}
        ${this.field("Rate kind", this.selectInput("exp-kind",
          Object.entries(this.state.catalog.rate_kinds), "mass"))}
        ${this.field("Unit written to the file", this.selectInput("exp-unit",
          this.state.units.groups.rate_mass.units.map((u) => [u.key, u.label]),
          Units.label.inputRate()))}
        ${this.field("Crop", this.selectInput("exp-crop",
          this.state.units.crops.map((c) => [c.key, c.label]), Units.get().crop))}
        <div class="row tight">
          ${this.field(`ISOXML cell (${lengthUnit})`,
            this.numberInput("exp-cell", Math.round(Units.convert.length(10)), "1", "1"))}
          ${this.field("Name", `<input type="text" id="exp-task"
            value="${this.escape(defaultName)}">`)}
        </div>
        <input type="hidden" id="exp-field" value="Field">
        <input type="hidden" id="exp-product" value="Product">
      </div>
      <div class="panel">
        <button class="primary wide" id="btn-run-export">Generate files</button>
      </div>

      ${this.qgisPanel()}`;

    document.getElementById("exp-source").addEventListener("change", (e) => {
      this.state.exportSource = e.target.value;
    });
    document.getElementById("exp-kind").addEventListener("change", () => {
      const kind = this.value("exp-kind");
      const group = kind === "volume" ? "rate_volume" : kind === "count" ? "rate_count" : "rate_mass";
      const select = document.getElementById("exp-unit");
      select.innerHTML = this.state.units.groups[group].units.map((u) =>
        `<option value="${u.key}">${this.escape(u.label)}</option>`).join("");
    });
    document.getElementById("btn-run-export").addEventListener("click", () => this.runExport());
    this.bindQgisPanel();
  },

  async runExport() {
    const formats = [
      ["shapefile", this.checked("fmt-shapefile")],
      ["isoxml", this.checked("fmt-isoxml")],
      ["csv", this.checked("fmt-csv")],
      ["geojson", this.checked("fmt-geojson")],
    ].filter(([, on]) => on).map(([f]) => f);

    if (!formats.length) { this.toast("Choose a format", "", "warn"); return; }

    const body = {
      formats,
      brand: this.value("exp-brand"),
      rate_property: "rate",
      rate_kind: this.value("exp-kind"),
      rate_unit: this.value("exp-unit"),
      crop: this.value("exp-crop"),
      cell_m: Units.toInternal.length(this.number("exp-cell", 10)),
      task_name: this.value("exp-task") || "Prescription",
      field_name: this.value("exp-field") || "Field",
      product_name: this.value("exp-product") || "Product",
    };

    if (this.value("exp-source") === "design") {
      if (!this.state.design) { this.toast("No trial", "Lay the trial out first.", "warn"); return; }
      body.features = this.state.design.features;
    } else {
      if (!this.state.selectedId) { this.toast("No data", "Pick a dataset.", "warn"); return; }
      body.dataset_id = this.state.selectedId;
      if (formats.includes("isoxml")) {
        this.toast("ISOXML needs polygons",
          "Use the package built from the trial layout to generate ISOXML.", "warn");
        return;
      }
    }

    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api("/api/export", { method: "POST", body }));
    if (!result) return;
    this.renderExportReport(result);
    this.toast("Files generated", `${result.bundle.entries.length} file(s) in the bundle.`);
  },

  renderExportReport(result) {
    const box = document.getElementById("export-report");
    if (!box) return;
    const rows = result.outputs.map((o) => {
      const detail = o.rate_field ? `rate field: ${o.rate_field}`
        : o.cols ? `${o.rows} x ${o.cols} grid, ${Units.num(o.cells_with_rate, 0)} cells with a rate`
        : o.features != null ? `${Units.num(o.features, 0)} features`
        : o.rows != null ? `${Units.num(o.rows, 0)} rows` : "";
      return `<tr><td>${this.escape(o.format)}</td><td>${this.escape(detail)}</td></tr>`;
    }).join("");
    const warnings = result.outputs.flatMap((o) => o.warnings || []);

    box.innerHTML = `
      <div class="panel">
        <h3>Files generated</h3>
        <a class="btn primary wide" style="display:block;text-decoration:none"
           href="${result.bundle.download_url}" download>
          Download ${this.escape(result.bundle.filename)}
        </a>
        <p class="hint tight">Written to:<br>
          <code style="font-size:11px">${this.escape(result.folder)}</code></p>
        <div class="scroll-x" style="margin-top:10px"><table class="data">
          <tr><th>Format</th><th>Detail</th></tr>${rows}
        </table></div>
        ${(result.notes || []).map((o) =>
          `<div class="note" style="margin-top:8px">${this.escape(o)}</div>`).join("")}
        ${warnings.map((w) =>
          `<div class="note warning" style="margin-top:6px">${this.escape(w)}</div>`).join("")}
      </div>`;
  },
});

/* ======================================================================
 * Import and dialogs
 * ==================================================================== */

Object.assign(App, {
  bindImport() {
    const input = document.getElementById("file-input");
    const folderInput = document.getElementById("folder-input");
    document.getElementById("btn-open-file").addEventListener("click", () => input.click());
    document.getElementById("btn-open-folder").addEventListener("click", () => folderInput.click());

    // Both pickers hand over File objects; the folder picker also knows each
    // file's place inside the chosen folder, which is what keeps a shapefile's
    // parts and a TASKDATA tree together on the server.
    for (const picker of [input, folderInput]) {
      picker.addEventListener("change", async () => {
        const files = Array.from(picker.files || []);
        picker.value = "";
        const file = files[0];
        if (!file) {
          // The file picker only comes back empty on cancel. The folder
          // picker also does on an empty folder, and the person who chose
          // it needs to hear that it was seen.
          if (picker === folderInput) this.toast("Nothing to import", "The folder holds no files.", "warn");
          return;
        }
        if (files.length === 1 && this.isProjectFile(file.name)) {
          await this.uploadProject(file);
          return;
        }
        await this.importFiles(files.map((f) => ({ file: f, path: f.webkitRelativePath || f.name })));
      });
    }

    document.getElementById("btn-open-path").addEventListener("click", () => {
      this.setPathDialogMode("data");
      document.getElementById("dlg-path").showModal();
      this.browse("");
    });

    for (const [id, kind] of [["btn-demo-harvest", "harvest"], ["btn-demo-trial", "trial"]]) {
      document.getElementById(id).addEventListener("click", async () => {
        const result = await this.busy(document.querySelector("aside.left"), () =>
          this.api("/api/import/demo", { method: "POST", body: { kind } }));
        if (!result) return;
        await this.refreshDatasets();
        await this.selectDataset(result.id);
      });
    }

    // The relief demo is a field, not an operation: it comes from the
    // terrain routes, which also carry the truth about what was put in it.
    document.getElementById("btn-demo-terrain").addEventListener("click", async () => {
      const result = await this.busy(document.querySelector("aside.left"), () =>
        this.api("/api/terrain/demo", { method: "POST" }));
      if (!result) return;
      await this.refreshDatasets();
      await this.selectDataset(result.id);
      this.goToTab("terrain");
    });

    /* Drag and drop anywhere on the window: files, a shapefile's parts, or a
     * whole folder.
     *
     * The listeners sit on the document, not on #app. While an import runs,
     * #app is dimmed with pointer events off, and a drag then lands on the
     * body — where, with nothing to cancel it, the browser's own answer to
     * a dropped file is to open it in place of the app, upload and all. The
     * same happens over an open dialog, which is not inside #app. Wherever a
     * file drag lands, its default is cancelled here; a drop that cannot be
     * taken right now is refused with a word rather than left to the browser.
     *
     * Only a file drag is ours. Text dragged from a table towards an input
     * keeps the browser's own drop, and does not light the outline that
     * promises an import. dragenter/dragleave fire for every element the
     * pointer crosses, so a depth count tells a real leave from a hop
     * between panels. */
    const dropZone = document.getElementById("app");
    const carriesFiles = (e) => {
      const types = Array.from(e.dataTransfer?.types || []);
      // A drag carrying nothing at all is neither text nor a link: an empty
      // folder from some file managers, or a synthetic event. Cancelling it
      // costs nothing, and the drop then gets its answer below.
      return !types.length || types.includes("Files");
    };
    // A drop is taken from the moment it lands. Reading a big folder's
    // entries takes a while, and the .busy class only went on once the
    // upload started: a second drop meanwhile passed the check below and
    // raced the first for the dataset list. The flag holds for the whole
    // drop; the class dims the window for the same span, and is the one
    // the pickers and every other request share.
    let reading = false;
    // Why a drop has to wait, or "" when it can be taken now.
    const blocked = () => {
      if (reading || document.querySelector(".busy")) {
        return "Something is still running. Wait for it to finish, then drop again.";
      }
      if (document.querySelector("dialog[open]")) return "Close the dialog first, then drop again.";
      return "";
    };
    let depth = 0;
    document.addEventListener("dragenter", (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      if (depth++ === 0 && !blocked()) dropZone.classList.add("dropping");
    });
    document.addEventListener("dragleave", (e) => {
      if (!carriesFiles(e)) return;
      if (--depth <= 0) { depth = 0; dropZone.classList.remove("dropping"); }
    });
    document.addEventListener("dragover", (e) => { if (carriesFiles(e)) e.preventDefault(); });
    document.addEventListener("drop", async (e) => {
      if (!carriesFiles(e)) return;
      e.preventDefault();
      depth = 0;
      dropZone.classList.remove("dropping");
      const why = blocked();
      if (why) { this.toast("Not now", why, "warn"); return; }
      reading = true;
      dropZone.classList.add("busy");
      try {
        const entries = await this.collectDropped(e.dataTransfer).catch((err) => {
          this.toast("Could not read the drop", `${err.message || err}\nTry "Open a folder" instead.`, "error");
          return [];
        });
        const file = entries[0]?.file;
        if (!file) {
          // An empty folder drops without a file in it; the person who dropped
          // it is otherwise left wondering whether the drop was seen.
          this.toast("Nothing to import", "The drop holds no files: an empty folder, or one " +
            "holding only hidden files.", "warn");
          return;
        }
        if (entries.length === 1 && this.isProjectFile(file.name)) { await this.uploadProject(file); return; }
        await this.importFiles(entries);
      } finally {
        reading = false;
        dropZone.classList.remove("busy");
      }
    });
  },

  /* The files in a drop, each with its path inside what was dropped.
   *
   * A folder only comes through DataTransferItem.webkitGetAsEntry, and that
   * call has to be made for every item before this function first awaits:
   * the item list is emptied the moment the drop event handler yields.
   * Without the entry API (or for an item that has none) the plain file list
   * is used, which carries loose files but no folders. */
  async collectDropped(transfer) {
    const items = Array.from(transfer?.items || []).filter((item) => item.kind === "file");
    const roots = items.map((item) => item.webkitGetAsEntry?.() ?? null);
    if (!roots.length || roots.some((entry) => !entry)) {
      return Array.from(transfer?.files || []).map((file) => ({ file, path: file.name }));
    }
    const found = [];
    const visit = async (entry) => {
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        found.push({ file, path: entry.fullPath.replace(/^\/+/, "") });
        return;
      }
      if (!entry.isDirectory) return;
      // readEntries hands out a folder in batches (a hundred at a time in
      // Chromium) and signals the end with an empty one.
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await visit(child);
      }
    };
    for (const root of roots) await visit(root);
    return found;
  },

  /* Everything picked or dropped goes up in one request, tree included: the
   * server can only read a shapefile's four files, or a TASKDATA folder, as
   * one thing when it sees them together. `entries` is [{file, path}] with
   * the path relative to what was dropped, forward slashes. */
  async importFiles(entries) {
    // The fold under Load data describes the last drop, and this is now the
    // last drop: a refusal below must not leave an older drop's list
    // standing as if it were this one's.
    this.renderDropReport([]);
    const maxFiles = 2000, maxBytes = 2 * 1024 ** 3;
    const bytes = entries.reduce((sum, entry) => sum + (entry.file.size || 0), 0);
    if (entries.length > maxFiles || bytes > maxBytes) {
      this.toast("Too much at once",
        `${Units.num(entries.length, 0)} file(s), ${this.fileSize(bytes)}. One drop takes up to ` +
        `${Units.num(maxFiles, 0)} files or 2 GB: drop one field's folder at a time, or zip it.`, "warn");
      return;
    }
    const form = new FormData();
    for (const { file, path } of entries) {
      form.append("files", file, file.name);
      form.append("paths", path);
    }
    // The whole window dims: a folder can take a while, and a second drop
    // meanwhile would only race the first for the dataset list.
    const result = await this.busy(document.getElementById("app"), () =>
      this.api("/api/import/files", { method: "POST", body: form }));
    if (!result) return;

    await this.refreshDatasets();
    await this.selectDataset(result.imported[0].id);
    this.renderDropReport(result.skipped);

    const skipped = result.skipped || [];
    // Thumbs.db, .DS_Store and their kind travel with the folder and the
    // server never walks them; named here, the file total still adds up.
    const ignored = result.ignored || 0;
    const lines = [`${Units.num(entries.length, 0)} file(s) → ${Units.num(result.imported.length, 0)} dataset(s)` +
      (ignored ? ` (${Units.num(ignored, 0)} system file(s) ignored)` : "") + "."];
    if (result.imported.length === 1) {
      const only = result.imported[0];
      lines[0] += ` ${only.meta.brand_label} · ${Units.num(only.rows, 0)} records.`;
    }
    for (const item of skipped.slice(0, 3)) lines.push(`Skipped ${item.name}: ${item.reason}`);
    if (skipped.length > 3) {
      lines.push(`…and ${skipped.length - 3} more, listed under "Load data".`);
    }
    this.toast(skipped.length ? "Imported, with files left out" : "Imported",
      lines.join("\n"), skipped.length ? "warn" : "");
  },

  /* What the last drop left out stays on the Load data panel until the next
   * one, or until the session is replaced: a toast is gone in five seconds,
   * and a missing layer would otherwise be looked for on the map. */
  renderDropReport(skipped) {
    const box = document.getElementById("drop-report");
    if (!skipped?.length) { box.innerHTML = ""; return; }
    box.innerHTML = `
      <details class="fold" open style="margin-top:8px">
        <summary>${skipped.length} file(s) skipped in the last drop</summary>
        <div class="inner">${skipped.map((item) =>
          `<p class="hint tight"><b style="font-family:var(--mono);font-size:11px">${
            this.escape(item.name)}</b><br>${this.escape(item.reason)}</p>`).join("")}
        </div>
      </details>`;
  },

  bindDialogs() {
    for (const button of document.querySelectorAll("dialog [data-close]")) {
      button.addEventListener("click", () => button.closest("dialog").close());
    }
    document.getElementById("btn-path-list").addEventListener("click",
      () => this.browse(this.value("path-input")));
    document.getElementById("btn-path-up").addEventListener("click",
      () => this.browse(this.state.browseParent || ""));
    document.getElementById("btn-path-open").addEventListener("click", async () => {
      const path = this.value("path-input");
      if (!path) return;
      // Choosing the projects folder: the same browser, used for the folder
      // itself rather than for something inside it. The server decides
      // whether it will do, so a refused folder leaves the picker open on it.
      if (this.state.folderPick) {
        await this.applyProjectsFolder(path);
        return;
      }
      // An .agrosuite file is a whole session, not one more layer: it goes to
      // "open", which replaces what is loaded, and never to import.
      if (this.state.projectOpen || this.isProjectFile(path)) {
        // Looked at before the picker closes: a refused path — a folder,
        // which a click in the browser puts in the box — leaves the picker
        // open on it, with the file inside still there to choose.
        const peek = await this.peekProject(path);
        if (!peek) return;
        document.getElementById("dlg-path").close();
        await this.openProject(path, peek);
        return;
      }
      // A .qgs or .qgz is a project, not a data file: it names layers rather
      // than holding them.
      if (this.state.qgisImport || /\.(qgs|qgz)$/i.test(path)) {
        this.state.qgisImport = false;
        document.getElementById("dlg-path").close();
        await this.importQgisProject(path);
        return;
      }
      const result = await this.busy(document.getElementById("dlg-path"), () =>
        this.api("/api/import/path", { method: "POST", body: { path } }));
      if (!result) return;
      document.getElementById("dlg-path").close();
      await this.refreshDatasets();
      await this.selectDataset(result.id);
      this.toast("Loaded",
        `${result.meta.brand_label} · ${Units.num(result.rows, 0)} records.`);
    });
    document.getElementById("btn-import-units-apply")
      .addEventListener("click", () => this.applyImportUnits());
  },

  async browse(path) {
    const payload = await this.api(`/api/browse?path=${encodeURIComponent(path || "")}`)
      .catch((err) => { this.toast("That did not work", err.message, "error"); return null; });
    if (!payload) return;

    this.state.browseParent = payload.parent;
    document.getElementById("path-current").textContent = payload.path;
    const box = document.getElementById("path-browser");
    box.innerHTML = "";

    if (!payload.entries.length) {
      // An empty folder is the normal case when one is being chosen to keep
      // projects in, and "nothing the app can open" would read as a refusal.
      box.innerHTML = this.state.folderPick
        ? '<div class="empty">This folder is empty. Press "Use this folder" to keep projects here.</div>'
        : '<div class="empty">Nothing the app can open in this folder.</div>';
      return;
    }
    for (const entry of payload.entries) {
      const node = document.createElement("div");
      node.className = "item";
      node.innerHTML = `<span class="icon">${entry.is_dir ? "📁" : "📄"}</span>
        <span class="name">${this.escape(entry.name)}</span>
        <span class="size">${entry.size != null ? this.fileSize(entry.size) : ""}</span>`;
      node.addEventListener("click", () => {
        document.getElementById("path-input").value = entry.path;
        if (entry.is_dir) this.browse(entry.path);
      });
      node.addEventListener("dblclick", () => {
        document.getElementById("path-input").value = entry.path;
        document.getElementById("btn-path-open").click();
      });
      box.appendChild(node);
    }
  },

  fileSize(bytes) {
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes, i = 0;
    while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
    return `${Units.num(value, i === 0 ? 0 : 1)} ${units[i]}`;
  },

  openUnitsDialog() {
    const prefs = Units.get();
    const groups = this.state.units.groups;
    const options = (group) => groups[group].units.map((u) => [u.key, u.label]);

    document.getElementById("units-body").innerHTML = `
      <p class="hint">Data is always stored in metric; these choices only change how
      the numbers appear and which unit files are written in.</p>
      ${this.field("Yield", this.selectInput("u-yield", options("rate_mass"), prefs.yield_unit))}
      ${this.field("Input rate", this.selectInput("u-input", options("rate_mass"), prefs.input_rate_unit))}
      ${this.field("Area", this.selectInput("u-area", options("area"), prefs.area_unit))}
      ${this.field("Length and width", this.selectInput("u-length", options("length"), prefs.length_unit))}
      ${this.field("Speed", this.selectInput("u-speed", options("speed"), prefs.speed_unit))}
      ${this.field("Mass", this.selectInput("u-mass", options("mass"), prefs.mass_unit))}
      ${this.field("Crop (bushel weight)", this.selectInput("u-crop",
        this.state.units.crops.map((c) => [c.key, `${c.label} — ${c.bushel_kg.toFixed(2)} kg/bu`]),
        prefs.crop))}
      ${this.field("Currency", this.selectInput("u-currency",
        this.state.units.currencies.map((c) => [c.key, c.label]), prefs.currency))}`;

    const apply = () => {
      Units.set("yield_unit", this.value("u-yield"));
      Units.set("input_rate_unit", this.value("u-input"));
      Units.set("area_unit", this.value("u-area"));
      Units.set("length_unit", this.value("u-length"));
      Units.set("speed_unit", this.value("u-speed"));
      Units.set("mass_unit", this.value("u-mass"));
      Units.set("crop", this.value("u-crop"));
      Units.set("currency", this.value("u-currency"));
      this.loadMap();
      this.renderTab();
      this.refreshPhrasing();
    };
    for (const id of ["u-yield", "u-input", "u-area", "u-length", "u-speed", "u-mass",
                      "u-crop", "u-currency"]) {
      document.getElementById(id).addEventListener("change", apply);
    }
    document.getElementById("dlg-units").showModal();
  },

  openImportUnitsDialog() {
    const groups = this.state.units.groups;
    const options = (group) => [["", "— leave as it is —"],
      ...groups[group].units.map((u) => [u.key, u.label])];

    document.getElementById("import-units-body").innerHTML = `
      ${this.field("Yield / rate in the file",
        this.selectInput("iu-rate", options("rate_mass"), ""))}
      ${this.field("Speed in the file", this.selectInput("iu-speed", options("speed"), ""))}
      ${this.field("Width, distance and altitude in the file",
        this.selectInput("iu-length", options("length"), ""))}
      ${this.field("Crop (for bushels)", this.selectInput("iu-crop",
        this.state.units.crops.map((c) => [c.key, c.label]), Units.get().crop))}`;
    document.getElementById("dlg-import-units").showModal();
  },

  /* One click instead of a dialog with three pickers. The app already worked
   * the units out from the file; asking the user to re-enter them would be
   * friction for its own sake. */
  async applyProposedUnits(proposed) {
    const source_units = {};
    if (proposed.rate) {
      source_units.value = proposed.rate;
      source_units.target_rate = proposed.rate;
      source_units.applied_rate = proposed.rate;
    }
    if (proposed.speed) source_units.speed_kmh = proposed.speed;
    if (proposed.length) {
      /* A monitor writes every length in the one system it was set to: the
       * GPS altitude is in feet whenever the width is, and the terrain
       * analyser reads it as metres unless told otherwise. */
      source_units.swath_m = proposed.length;
      source_units.distance_m = proposed.length;
      source_units.elev_m = proposed.length;
    }

    const result = await this.busy(document.getElementById("right-panel"), () =>
      this.api(`/api/datasets/${this.state.selectedId}/units`, {
        method: "POST",
        body: { source_units, crop: proposed.crop || Units.get().crop },
      }));
    if (!result) return;

    await this.refreshDatasets();
    await this.selectDataset(result.dataset.id);
    this.toast("Units applied", result.conversions.join("\n"));
  },

  async applyImportUnits() {
    const source_units = {};
    const rate = this.value("iu-rate");
    if (rate) {
      source_units.value = rate;
      source_units.target_rate = rate;
      source_units.applied_rate = rate;
    }
    if (this.value("iu-speed")) source_units.speed_kmh = this.value("iu-speed");
    if (this.value("iu-length")) {
      source_units.swath_m = this.value("iu-length");
      source_units.distance_m = this.value("iu-length");
      source_units.elev_m = this.value("iu-length");
    }
    if (!Object.keys(source_units).length) {
      this.toast("Nothing to convert", "Choose at least one unit.", "warn");
      return;
    }

    const result = await this.busy(document.getElementById("dlg-import-units"), () =>
      this.api(`/api/datasets/${this.state.selectedId}/units`, {
        method: "POST",
        body: { source_units, crop: this.value("iu-crop") },
      }));
    if (!result) return;
    document.getElementById("dlg-import-units").close();
    await this.refreshDatasets();
    await this.selectDataset(result.dataset.id);
    this.toast("Units converted", result.conversions.join("\n"));
  },
});

document.addEventListener("DOMContentLoaded", () => App.init());

/* ======================================================================
 * The guided workflow
 *
 * Two tracks, and the stages of whichever one the project is on: always
 * visible, with what is missing and the single next thing worth doing. The
 * stage is derived on the server from what has actually happened, so it can
 * never disagree with the state of the session — and it guides rather than
 * gates: every stage in the strip is clickable, whatever state it is in.
 * ==================================================================== */

Object.assign(App, {
  async refreshProject() {
    const project = await this.api("/api/project").catch(() => null);
    if (!project) return;
    this.state.project = project;
    this.renderStageStrip(project);
    this.renderProjectPanel(project);
  },

  renderStageStrip(project) {
    const strip = document.getElementById("stage-strip");
    if (!strip) return;
    const tracks = this.state.catalog.workflow.tracks;
    const stages = project.stages || [];
    const empty = !this.state.datasets.length;
    const next = project.next_action;
    strip.hidden = false;

    // The choice between the two tracks comes first, because it is the one
    // decision that changes what the rest of the strip is for.
    const choice = Object.entries(tracks).map(([key, track]) => `
      <button class="track${key === project.track ? " on" : ""}" data-track="${key}"
              title="${this.escape(track.description)}">${this.escape(track.label)}</button>`
    ).join("");

    // Done, skipped or still ahead — none of them closed. A stage nobody has
    // reached is as clickable as the one the project is standing on.
    const marks = { done: "✓", current: "●", skippable: "–", ahead: "○" };
    const drawn = stages.map((stage) => `
      <button class="stage ${stage.state}" data-stage="${stage.key}"
              title="${this.escape(stage.description)} — If you stop here: ${
                this.escape(stage.produces)}. ${this.escape(stage.export)}">
        <span class="dot">${marks[stage.state] || "○"}</span>
        <span>${this.escape(stage.label)}</span>
      </button>`).join("");

    strip.innerHTML = `
      <div class="tracks" role="group" aria-label="What are you doing?">${choice}</div>
      ${drawn}
      <div class="next">
        ${empty ? `<span class="what">Either way, start by opening a file: a boundary
          to plan a trial on, or the harvest files to evaluate one.</span>`
          : `<span class="what">Next: ${this.escape(next.label)}</span>`}
        <button class="small" id="btn-next-action">Go</button>
        ${next.stop && !empty ? `<span class="stop" title="${
          this.escape(next.stop.produces)}. ${
          this.escape(next.stop.export)}">or stop here</span>` : ""}
      </div>`;

    for (const button of strip.querySelectorAll("[data-track]")) {
      button.addEventListener("click", () => this.setTrack(button.dataset.track));
    }
    for (const button of strip.querySelectorAll("[data-stage]")) {
      button.addEventListener("click", () => this.goToStep(button.dataset.stage));
    }
    document.getElementById("btn-next-action").addEventListener("click",
      () => this.goToStep(next.step));
    document.getElementById("btn-next-action").title = next.why;
  },

  /* The two-way choice sets the project's goal, which is what the server
   * reads: planning a trial and evaluating one need different things, and
   * the panel on the left should ask for the ones that apply. */
  async setTrack(track) {
    const goal = this.state.catalog.workflow.tracks[track]?.goal;
    if (!goal || goal === this.state.project?.goal) return;
    await this.api("/api/project", { method: "POST", body: { goal } }).catch((err) => {
      this.toast("That did not work", err.message, "error");
    });
    await this.refreshProject();
    this.renderTab();
  },

  /* Each workflow step maps to the tab that carries it out. */
  goToStep(step) {
    const tab = {
      load: "dados", review: "dados", units: "dados", columns: "dados",
      clean: "limpeza", design: "ensaio", analyse: "difm", augmenta: "dados",
      terrain: "terrain", export: "exportar",
    }[step] || "dados";
    this.goToTab(tab);
  },

  renderProjectPanel(project) {
    const box = document.getElementById("project-panel");
    if (!box) return;
    const goals = this.state.catalog.workflow.goals;
    const evaluation = project.evaluation;

    const layers = project.layers.length ? project.layers.map((layer) => `
      <div class="layer-row">
        <span class="name" title="${this.escape(layer.label)}">${this.escape(layer.label)}</span>
        <select data-role-for="${layer.dataset_id}">
          ${Object.entries(project.role_options).map(([key, label]) =>
            `<option value="${key}"${key === layer.role ? " selected" : ""}>${
              this.escape(label)}</option>`).join("")}
        </select>
      </div>`).join("")
      : '<div class="empty" style="padding:12px">No file has a role yet.</div>';

    const requirement = (r) => `
      <div class="req ${r.satisfied ? "done" : "todo"}">
        <span class="mark">${r.satisfied ? "✓" : "○"}</span>
        <span class="txt"><b>${this.escape(r.label)}</b>
          ${r.satisfied ? "" : `<br><span class="d">${this.escape(r.detail)}</span>`}</span>
      </div>`;

    box.innerHTML = `
      ${this.field("Goal", this.selectInput("project-goal",
        Object.entries(goals).map(([k, v]) => [k, v.label]), project.goal))}
      <h4 style="margin:10px 0 6px">Files and their roles</h4>
      ${layers}
      ${project.goal === "difm" ? this.priceFields(project) : ""}
      <h4 style="margin:12px 0 4px">What this needs</h4>
      ${evaluation.requirements.map(requirement).join("")}
      ${evaluation.optional.length ? `<details class="fold" style="margin-top:8px">
        <summary>Optional, but worth having</summary>
        <div class="inner">${evaluation.optional.map(requirement).join("")}</div>
      </details>` : ""}
      ${evaluation.ready && project.roles_present.length > 1 ? `
        <button class="wide" id="btn-join-layers" style="margin-top:10px">
          Join the layers for analysis
        </button>
        <p class="hint tight">One file per layer, three passes over the same ground.
        The join puts them on a shared grid so the response can be measured.</p>` : ""}`;

    for (const id of ["price-crop", "price-value", "price-cost"]) {
      document.getElementById(id)?.addEventListener("change", () => this.savePrices());
    }

    document.getElementById("project-goal").addEventListener("change", async (event) => {
      await this.api("/api/project", { method: "POST", body: { goal: event.target.value } });
      this.refreshProject();
      this.renderTab();
    });
    for (const select of box.querySelectorAll("[data-role-for]")) {
      select.addEventListener("change", async (event) => {
        await this.busy(box, () => this.api("/api/project/role", {
          method: "POST",
          body: { dataset_id: event.target.dataset.roleFor, role: event.target.value },
        }));
        await this.refreshDatasets();
        this.refreshProject();
      });
    }
    document.getElementById("btn-join-layers")?.addEventListener("click", () => this.joinLayers());
  },

  /* Prices belong to the project, not to one analysis: the same crop price
   * decides the optimum, the profit curve and the value of the whole trial. */
  priceFields(project) {
    const prices = project.prices || {};
    const yieldUnit = Units.label.yield();
    const inputUnit = Units.label.inputRate();
    const symbol = Units.currencySymbol();
    const crop = prices.crop || Units.get().crop;

    // Stored per internal unit (per kg); shown per selling unit (per bushel,
    // per pound), which is how prices are actually quoted.
    const shownCrop = prices.crop_price
      ? Units.priceFromInternal(prices.crop_price, yieldUnit) : "";
    const shownInput = prices.input_cost
      ? Units.priceFromInternal(prices.input_cost, inputUnit) : "";

    return `
      <h4 style="margin:12px 0 6px">Prices</h4>
      ${this.field("Crop", this.selectInput("price-crop",
        this.state.units.crops.map((c) => [c.key, c.label]), crop))}
      <div class="row tight">
        ${this.field(`${symbol}/${Units.rateNumerator(yieldUnit)}`,
          `<input type="number" id="price-value" step="0.01" min="0"
             value="${shownCrop === "" ? "" : Number(shownCrop).toFixed(2)}"
             placeholder="crop">`)}
        ${this.field(`${symbol}/${Units.rateNumerator(inputUnit)}`,
          `<input type="number" id="price-cost" step="0.01" min="0"
             value="${shownInput === "" ? "" : Number(shownInput).toFixed(2)}"
             placeholder="input">`)}
      </div>`;
  },

  async savePrices() {
    const crop = this.value("price-crop");
    if (crop) Units.set("crop", crop);
    const body = {
      crop_price: Units.priceToInternal(this.number("price-value", 0), Units.label.yield()),
      input_cost: Units.priceToInternal(this.number("price-cost", 0), Units.label.inputRate()),
      currency: Units.get().currency,
      crop,
    };
    await this.api("/api/project/prices", { method: "POST", body }).catch(() => null);
    this.refreshProject();
  },

  async joinLayers() {
    /* The zone column is whatever the yield layer carries beyond the canonical
     * columns — it is what lets the analysis split the field instead of
     * pooling two different responses into one curve. */
    const carry = [];
    const yieldLayer = this.state.project?.layers.find((l) => l.role === "yield");
    if (yieldLayer) {
      const detail = await this.api(`/api/datasets/${yieldLayer.dataset_id}`).catch(() => null);
      for (const column of detail?.columns || []) {
        if (this.state.catalog.columns[column] || ["x", "y"].includes(column)) continue;
        carry.push(column);
      }
    }

    const result = await this.busy(document.querySelector("aside.left"), () =>
      this.api("/api/project/join", { method: "POST", body: { carry, min_purity: 0.8 } }));
    if (!result) return;

    await this.refreshDatasets();
    await this.selectDataset(result.dataset.id);
    this.refreshProject();
    this.state.joinReport = result.report;

    const lengthUnit = Units.label.length();
    this.toast(
      "Layers joined",
      `${result.report.cells} cells at ${Units.num(Units.convert.length(result.cell_m), 0)} ` +
      `${lengthUnit}. ${result.report.notes[0] || ""}`,
    );
    this.goToStep("analyse");
  },
});

/* ======================================================================
 * Machine profiles
 *
 * A header width, a flow delay and a working speed belong to the machine,
 * not to the field, yet three tabs ask for them. A profile is entered once,
 * kept by the server in profiles.json, and picked from one small row on each
 * tab: choosing a machine fills the fields that tab needs, "Save as
 * machine" reads them back, "Update machine" edits the one picked. The
 * server owns the list, so it is fetched on every render; the client keeps
 * only which machine each tab picked, so a tab switch does not forget it.
 * Every number crosses the wire in metric and is shown in the units the
 * user chose.
 * ==================================================================== */

Object.assign(App, {
  /* What each tab takes from a profile and gives back to one. `fill` puts
   * the profile's values on the form and says what it did, in the user's
   * units; `collect` reads the form back, in metric, leaving out what the
   * form does not have so the dialog can ask for it. */
  machineScope(scope) {
    const shown = (kind, value) => Number(Units.convert[kind](value).toFixed(3));
    const said = (value) => String(Number(Number(value).toFixed(2)));
    const put = (id, value) => {
      const node = document.getElementById(id);
      if (node) node.value = value;
    };
    const metric = (kind, id) => this.machineFieldMetric(kind, id) ?? undefined;
    const kindFromOperation = (d) => ({
      harvest: "combine", planting: "seeder", application: "sprayer",
    })[d?.meta?.operation] || null;

    const scopes = {
      clean: {
        hint: "A saved machine fills the flow delay and the speed range.",
        kind: (d) => kindFromOperation(d) || "other",
        fill: (p) => {
          const speedUnit = Units.label.speed();
          const low = shown("speed", p.speed_min_kmh);
          const high = shown("speed", p.speed_max_kmh);
          put("clean-delay", p.flow_delay_s);
          put("p-speed_range-min", low);
          put("p-speed_range-max", high);
          this.machinePlaced("p-speed_range-min", low, p.speed_min_kmh);
          this.machinePlaced("p-speed_range-max", high, p.speed_max_kmh);
          // Filling a filter that is switched off would change nothing and say
          // nothing; switching it on is what picking a speed range means.
          const step = document.getElementById("en-speed_range");
          const switched = step && !step.checked;
          if (switched) step.checked = true;
          return `flow delay ${said(p.flow_delay_s)} s, speed ${said(low)}–` +
            `${said(high)} ${speedUnit}.` +
            (switched ? " The speed filter was switched on." : "");
        },
        collect: () => ({
          flow_delay_s: this.number("clean-delay", 0),
          speed_min_kmh: metric("speed", "p-speed_range-min"),
          speed_max_kmh: metric("speed", "p-speed_range-max"),
        }),
      },
      design: {
        hint: "A saved machine fills the implement width and the passes per strip.",
        // A yield map is usually what is selected, but the trial is laid out
        // for whatever applies the rates.
        kind: (d) => (d?.meta?.operation === "harvest" ? "seeder"
          : kindFromOperation(d) || "seeder"),
        fill: (p) => {
          const width = shown("length", p.implement_width_m);
          put("design-width", width);
          put("design-passes", p.passes_per_strip);
          this.machinePlaced("design-width", width, p.implement_width_m);
          return `implement width ${said(width)} ${Units.label.length()}, ` +
            `${p.passes_per_strip} pass(es) per strip.`;
        },
        collect: () => ({
          implement_width_m: metric("length", "design-width"),
          passes_per_strip: this.number("design-passes", 2),
        }),
      },
      package: {
        hint: "A saved machine picks the target monitor.",
        kind: (d) => kindFromOperation(d) || "other",
        fill: (p) => {
          const select = document.getElementById("pkg-monitor");
          const known = [...select.options].some((o) => o.value === p.monitor);
          select.value = known ? p.monitor : "generic";
          // The existing handler redraws what the platform accepts.
          select.dispatchEvent(new Event("change"));
          const label = select.options[select.selectedIndex]?.textContent || select.value;
          return known ? `platform ${label}.`
            : `there is no package layout for ${p.monitor_label}, so the platform is ${label}.`;
        },
        collect: () => ({ monitor: this.value("pkg-monitor") || "generic" }),
      },
    };
    return scopes[scope];
  },

  /* One select and two small buttons; the "Suggest from this file" step
   * lives inside the save dialog, where its result can be checked before
   * anything is written. */
  machinePicker(scope) {
    const { hint } = this.machineScope(scope);
    const picked = this.machinePickedIn(scope);
    return `
      <div class="row tight machine-row">
        <label class="field">
          <span>Machine</span>
          <select id="machine-${scope}" data-machine-picker>${
            this.machineOptions(picked?.name || "")}</select>
        </label>
        <button class="small" id="btn-machine-save-${scope}"
          title="Save the settings on this tab as a machine">${
            this.machineSaveLabel(picked)}</button>
        <button class="small ghost" id="btn-machine-delete-${scope}"
          title="Delete the chosen machine">✕</button>
      </div>
      <p class="hint tight">${hint}</p>`;
  },

  machineOptions(selected = "") {
    const profiles = this.state.machines?.profiles || [];
    const placeholder = this.state.machinesRefused ? "— profiles.json could not be read —"
      : profiles.length ? "— choose a machine —" : "— no machine saved yet —";
    // A profile typed into the file that the app would refuse to save is
    // listed, so it can be fixed from here, but marked.
    const label = (p) => `${p.name} · ${p.kind_label}` +
      (p.problems?.length ? " · ⚠ needs attention" : "");
    return [["", placeholder], ...profiles.map((p) => [p.name, label(p)])]
      .map(([value, text]) =>
        `<option value="${this.escape(value)}"${value === selected ? " selected" : ""}>${
          this.escape(text)}</option>`).join("");
  },

  /* With a machine picked, the save button edits it rather than adding one. */
  machineSaveLabel(picked) {
    return picked ? "Update machine…" : "Save as machine…";
  },

  bindMachinePicker(scope) {
    const select = document.getElementById(`machine-${scope}`);
    if (!select) return;
    select.addEventListener("change", () => this.pickMachine(scope, select.value));
    document.getElementById(`btn-machine-save-${scope}`)
      .addEventListener("click", () => this.openMachineDialog(scope));
    document.getElementById(`btn-machine-delete-${scope}`)
      .addEventListener("click", () => this.deleteMachine(scope));
    // The tab was just redrawn from its defaults — a tab switch, a unit
    // change — over whatever the machine picked here had filled in. Its
    // values go back, quietly: the picker shows which machine they are.
    const picked = this.machinePickedIn(scope);
    if (picked) this.applyMachine(scope, picked, { quiet: true });
    this.loadMachines();
  },

  bindMachineDialog() {
    document.getElementById("btn-machine-suggest")
      .addEventListener("click", () => this.suggestMachine());
    document.getElementById("btn-machine-save")
      .addEventListener("click", () => this.saveMachine());
  },

  /* The server owns the list: it is fetched on every render rather than kept
   * on the client, so a profile added in another window, or by hand in the
   * file, shows up at the next tab change. `always` repeats a refusal that
   * was already said: a click asked for the list and gets its answer. */
  async loadMachines({ always = false } = {}) {
    const listing = await this.api("/api/profiles").catch((err) => {
      // A profiles.json the app cannot read is refused, not overwritten: the
      // message names the file and what is wrong. Every tab asks for the list
      // again, and the same refusal is said once, not at every tab switch.
      if (always || this.state.machinesRefused !== err.message) {
        this.toast("Machine profiles unavailable", err.message, "error");
      }
      this.state.machinesRefused = err.message;
      return null;
    });
    if (listing) this.state.machinesRefused = null;
    this.setMachines(listing);
    return listing;
  },

  setMachines(listing) {
    this.state.machines = listing;
    for (const select of document.querySelectorAll("select[data-machine-picker]")) {
      this.drawMachinePicker(select.id.replace("machine-", ""));
    }
  },

  /* The picker's list and its save button, in line with the list the server
   * holds now and with what this tab picked. */
  drawMachinePicker(scope) {
    const select = document.getElementById(`machine-${scope}`);
    if (!select) return;
    const picked = this.machinePickedIn(scope);
    select.innerHTML = this.machineOptions(picked?.name || "");
    const button = document.getElementById(`btn-machine-save-${scope}`);
    if (button) button.textContent = this.machineSaveLabel(picked);
  },

  /* What the tab picked, remembered by name so the choice survives the tab
   * being redrawn; resolved against the list as it is now, so a machine
   * deleted in the meantime is simply no longer picked. */
  machinePickedIn(scope) {
    return this.machineNamed(this.state.machinePicked?.[scope]);
  },

  pickMachine(scope, name) {
    this.state.machinePicked = { ...(this.state.machinePicked || {}), [scope]: name };
    const profile = this.machineNamed(name);
    document.getElementById(`btn-machine-save-${scope}`).textContent =
      this.machineSaveLabel(profile);
    if (profile) this.applyMachine(scope, profile);
  },

  /* The machine's values onto the tab, with what was done said in a toast.
   * A profile the file holds but the app would refuse to save — an inverted
   * speed range, a kind it does not know — is not applied: its numbers would
   * go into the cleaning as they are. It stays picked, so "Update machine…"
   * opens on it and can put it right. */
  applyMachine(scope, profile, { quiet = false } = {}) {
    if (profile.problems?.length) {
      if (!quiet) {
        this.toast(`'${profile.name}' needs attention`,
          `${profile.problems.join(" ")} Fix it with "Update machine…", or in ` +
          `${this.state.machines.storage}. Nothing was filled in.`, "warn");
      }
      return;
    }
    const done = this.machineScope(scope).fill(profile);
    if (!quiet) this.toast(`Filled from '${profile.name}'`, done);
  },

  /* Names match the way the server matches them — case and spacing aside —
   * so the profile just saved is found under whatever spelling came back. */
  machineKey(name) {
    return String(name || "").split(/\s+/).join(" ").trim().toLowerCase();
  },

  machineNamed(name) {
    const key = this.machineKey(name);
    if (!key) return null;
    return (this.state.machines?.profiles || []).find((p) => this.machineKey(p.name) === key)
      || null;
  },

  machineKindLabel(kind) {
    return Object.fromEntries(this.state.machines?.kinds || [])[kind] || kind;
  },

  /* A typed value on its way to the server, in metric. Rounded to a
   * micrometre, not a millimetre: 33 ft is 10.0584 m, and cut to 10.058 it
   * came back on screen as 32.999 while the toast still said 33. Six
   * decimals keep profiles.json readable and are invisible at the three the
   * forms show. */
  machineMetric(kind, value) {
    return Math.round(Units.toInternal[kind](value) * 1e6) / 1e6;
  },

  /* What the app itself put in a physical field, with the metric number it
   * was converted from. A field still showing that value is read back as
   * that number rather than re-converted: 9 m shown as 29.528 ft and
   * converted back is 9.000134 m, a drift that would be written to
   * profiles.json and read as a change nobody made. Keyed by field id; the
   * ids are distinct across the tabs and the dialog. */
  machinePlaced(id, display, metric) {
    const placed = (this.state.machinePlaced = this.state.machinePlaced || {});
    if (metric == null || display === "") delete placed[id];
    else placed[id] = { display, metric };
  },

  /* A physical field read back in metric: the exact number the app placed
   * there while the field still shows it, the typed value converted once it
   * does not. Null when the field is empty. */
  machineFieldMetric(kind, id) {
    const value = this.number(id, null);
    if (value == null) return null;
    const placed = this.state.machinePlaced?.[id];
    return placed && placed.display === value ? placed.metric : this.machineMetric(kind, value);
  },

  /* "John Deere combine", or "My seeder" when the file does not say who made it. */
  machineNameFor(d, kind) {
    const brand = (d?.meta?.brand_label || "").trim();
    const known = brand && !/^(desconhecido|unknown|generic)/i.test(brand) ? brand : "My";
    const word = kind === "other" ? "machine"
      : this.machineKindLabel(kind).split(" /")[0].toLowerCase();
    return `${known} ${word}`;
  },

  /* The server names a suggestion with its width in metres; whoever reads
   * feet should see feet, rounded the way a header is spoken of. */
  machineNameWithWidth(p) {
    const base = p.name.replace(/\s+[\d.]+\s*m$/, "");
    if (!(p.implement_width_m > 0)) return base;
    const unit = Units.label.length();
    const width = Units.convert.length(p.implement_width_m);
    return `${base} ${Number(width.toFixed(unit === "m" ? 2 : 0))} ${unit}`;
  },

  async openMachineDialog(scope) {
    if (!this.state.machines && !(await this.loadMachines({ always: true }))) return;
    const d = this.state.selected;
    const handlers = this.machineScope(scope);
    // A picked machine is edited, not copied: the dialog opens on its name,
    // kind, monitor and notes, and saving under that name updates it without
    // asking. Otherwise the form starts a new one from the file.
    const picked = this.machinePickedIn(scope);
    const kind = picked?.kind || handlers.kind(d);
    const brands = this.state.catalog.brands.map((b) => b.key);
    const draft = picked ? { ...picked } : {
      name: this.machineNameFor(d, kind),
      kind,
      monitor: brands.includes(d?.meta?.brand) ? d.meta.brand : "generic",
      implement_width_m: null,
      flow_delay_s: kind === "combine" ? 12 : 0,
      passes_per_strip: 2,
      speed_min_kmh: null,
      speed_max_kmh: null,
      notes: "",
    };
    // What the tab shows wins over what was saved: a width corrected on the
    // form is the correction being saved. A field the machine filled and
    // nobody touched reads back as the exact number it came from.
    for (const [key, value] of Object.entries(handlers.collect())) {
      if (value !== undefined) draft[key] = value;
    }
    this.state.machineDialog = { scope, editing: picked?.name || null };
    this.renderMachineDialog(draft);
    document.getElementById("dlg-machine").showModal();
  },

  renderMachineDialog(p) {
    const lengthUnit = Units.label.length();
    const speedUnit = Units.label.speed();
    const editing = this.state.machineDialog?.editing;
    // A speed minimum of 0 is a real setting and stays 0; an empty box is
    // what makes the server fill in the kind's typical range instead.
    const shown = (kind, value) => (value == null ? "" : Number(Units.convert[kind](value).toFixed(3)));
    const brands = this.state.catalog.brands.map((b) => [b.key, b.label]);
    // Only the width keeps 0 for "unknown": that is what a suggestion sends
    // when the file has no swath, and the placeholder is the better prompt.
    const width = shown("length", p.implement_width_m || null);
    const low = shown("speed", p.speed_min_kmh);
    const high = shown("speed", p.speed_max_kmh);
    this.machinePlaced("mp-width", width, p.implement_width_m || null);
    this.machinePlaced("mp-speed-min", low, p.speed_min_kmh);
    this.machinePlaced("mp-speed-max", high, p.speed_max_kmh);

    document.querySelector("#dlg-machine h3").textContent =
      editing ? `Update '${editing}'` : "Save as a machine";
    document.getElementById("machine-body").innerHTML = `
      ${this.field("Name",
        `<input type="text" id="mp-name" value="${this.escape(p.name)}"
           placeholder="e.g. S780 with the 40 ft draper">`,
        editing ? `Under another name this is saved as a new machine; '${this.escape(editing)}' stays.`
          : "Saving under another machine's name asks before replacing it.")}
      <div class="row tight">
        ${this.field("Kind", this.selectInput("mp-kind", this.state.machines.kinds, p.kind))}
        ${this.field("Monitor", this.selectInput("mp-monitor", brands, p.monitor))}
      </div>
      <div class="row tight">
        ${this.field(`Implement width (${lengthUnit})`,
          `<input type="number" id="mp-width" value="${width}"
             step="any" min="0" placeholder="header, boom or drill">`)}
        ${this.field("Flow delay (s)", this.numberInput("mp-delay", p.flow_delay_s, "0.5", "0"),
          "Combines only; 0 for anything else.")}
      </div>
      <div class="row tight">
        ${this.field(`Speed from (${speedUnit})`,
          `<input type="number" id="mp-speed-min" value="${low}"
             step="any" min="0" placeholder="typical for the kind">`)}
        ${this.field(`to (${speedUnit})`,
          `<input type="number" id="mp-speed-max" value="${high}"
             step="any" min="0" placeholder="typical for the kind">`)}
        ${this.field("Passes per strip", this.numberInput("mp-passes", p.passes_per_strip, "1", "1"))}
      </div>
      ${this.field("Notes", `<input type="text" id="mp-notes" value="${this.escape(p.notes)}">`)}
      <div id="mp-problems"></div>
      <p class="hint tight">Saved in <span style="font:11px var(--mono)">${
        this.escape(this.state.machines.storage)}</span> — plain JSON, in metric.</p>`;

    document.getElementById("btn-machine-suggest").hidden = !this.state.selectedId;
  },

  /* Prefill from the selected dataset. Nothing is saved: the file is
   * evidence of what the machine was, and the operator confirms it. */
  async suggestMachine() {
    if (!this.state.selectedId) {
      this.toast("No file selected", "Pick a dataset on the left first.", "warn");
      return;
    }
    // The suggestion is for the kind the form shows, not for whichever
    // machine wrote the file: on the trial-layout tab a yield map is
    // selected, and the profile being made is the seeder's.
    const kind = this.value("mp-kind");
    const passes = this.value("mp-passes");
    const result = await this.busy(document.getElementById("dlg-machine"), () =>
      this.api("/api/profiles/suggest", {
        method: "POST", body: { dataset_id: this.state.selectedId, kind },
      }));
    if (!result) return;

    const profile = { ...result.profile, name: this.machineNameWithWidth(result.profile) };
    // Passes per strip are a decision, not something a file records.
    if (passes) profile.passes_per_strip = passes;
    // Suggesting into a machine being updated fills in its numbers, not a
    // new name: the file says what the machine is like, not what it is called.
    if (this.state.machineDialog?.editing) profile.name = this.value("mp-name");
    this.renderMachineDialog(profile);
    document.getElementById("mp-problems").innerHTML = result.problems.length
      ? `<div class="note warning">${result.problems.map((s) => this.escape(s)).join("<br>")}</div>`
      : `<div class="note ok">Read from '${this.escape(result.dataset_label)}'. ` +
        "Nothing stops this being saved as it is.</div>";
    this.toast("Suggested from the file",
      result.problems.length ? "Fill in what is missing, then save." : "Check the numbers, then save.");
  },

  async saveMachine() {
    const { scope, editing } = this.state.machineDialog || {};
    const name = (this.value("mp-name") || "").trim();

    // Checked here first, in the units on the screen: the server refuses an
    // inverted range too, but in km/h, quoting numbers nobody typed.
    const low = this.number("mp-speed-min", null);
    const high = this.number("mp-speed-max", null);
    if (low != null && high != null && low >= high) {
      this.toast("Check the speed range", "The minimum speed must be below the maximum " +
        `(got ${low} and ${high} ${Units.label.speed()}).`, "warn");
      return;
    }

    const body = {
      name,
      kind: this.value("mp-kind"),
      monitor: this.value("mp-monitor"),
      // A missing width goes as 0 so the server refuses it with its own words
      // rather than inventing a header.
      implement_width_m: this.machineFieldMetric("length", "mp-width") ?? 0,
      flow_delay_s: this.number("mp-delay", 0),
      passes_per_strip: this.number("mp-passes", 2),
      // Left empty, the speeds come back as the typical range for the kind.
      speed_min_kmh: this.machineFieldMetric("speed", "mp-speed-min"),
      speed_max_kmh: this.machineFieldMetric("speed", "mp-speed-max"),
      notes: this.value("mp-notes") || "",
      // Updating the machine that was picked is what the dialog opened for
      // and asks nothing; any other saved name is replaced only once the
      // person says so, below.
      replace: !!editing && this.machineKey(name) === this.machineKey(editing),
    };

    // A 400 lists every problem at once; busy() shows it verbatim. A 409 is
    // a question, with the saved machine's name in it.
    const listing = await this.busy(document.getElementById("dlg-machine"), async () => {
      try {
        return await this.api("/api/profiles", { method: "POST", body });
      } catch (err) {
        if (err.status !== 409) throw err;
        if (!window.confirm(`${err.message}\n\nReplace that machine with what the form shows?`)) {
          return null;
        }
        return this.api("/api/profiles", { method: "POST", body: { ...body, replace: true } });
      }
    });
    if (!listing) return;

    document.getElementById("dlg-machine").close();
    this.setMachines(listing);
    const saved = this.machineNamed(name);
    let filled = "";
    if (saved && document.getElementById(`machine-${scope}`)) {
      this.state.machinePicked = { ...(this.state.machinePicked || {}), [scope]: saved.name };
      this.drawMachinePicker(scope);
      filled = " " + this.machineScope(scope).fill(saved);
    }
    this.toast(editing ? "Machine updated" : "Machine saved",
      `'${saved?.name || name}' is ready on every tab.${filled}`);
  },

  async deleteMachine(scope) {
    const name = this.value(`machine-${scope}`);
    if (!name) {
      this.toast("Nothing to delete", "Choose a machine in the list first.", "warn");
      return;
    }
    if (!window.confirm(`Delete the machine '${name}'? Its saved settings are lost; ` +
                        "what is already filled in on this tab stays as it is.")) return;
    const listing = await this.busy(document.getElementById(`machine-${scope}`).parentElement,
      () => this.api(`/api/profiles/${encodeURIComponent(name)}`, { method: "DELETE" }));
    if (!listing) return;
    // Forgotten on every tab, not just this one: a machine saved later under
    // the same name is a different machine and is not picked by itself.
    for (const [tab, picked] of Object.entries(this.state.machinePicked || {})) {
      if (this.machineKey(picked) === this.machineKey(name)) this.state.machinePicked[tab] = "";
    }
    this.setMachines(listing);
    this.toast("Machine deleted", `'${name}' is gone from the list.`);
  },
});

/* ======================================================================
 * Project files
 *
 * A session is worth keeping: the datasets, the cleaning results, the
 * roles and the prices took an afternoon to put together, and the app
 * forgets all of it when it closes. "Save project" writes one .agrosuite
 * file holding the whole session; "Open project" brings it back as it
 * was, with the same dataset ids, so every report still points at its
 * datasets. The server owns the recent list; the client only asks for it.
 * ==================================================================== */

Object.assign(App, {
  bindProjectFiles() {
    document.getElementById("btn-project-save")
      .addEventListener("click", () => this.openSaveDialog());
    document.getElementById("btn-save-project-confirm")
      .addEventListener("click", () => this.saveProject());
    for (const id of ["save-name", "save-path"]) {
      document.getElementById(id).addEventListener("keydown", (event) => {
        if (event.key === "Enter") { event.preventDefault(); this.saveProject(); }
      });
    }
    document.getElementById("btn-project-new")
      .addEventListener("click", () => this.newProject());

    // The same browser as "Open by path", pointed at projects: it lists
    // .agrosuite files already, and an .agrosuite pick goes to "open".
    document.getElementById("btn-project-open").addEventListener("click", () => {
      this.setPathDialogMode("project");
      document.getElementById("path-input").value = "";
      document.getElementById("dlg-path").showModal();
      const last = (this.state.recent || []).find((item) => item.exists);
      this.browse(last ? last.path : "");
      this.toast("Pick a project", "Choose a .agrosuite file, then press Open.");
    });
    // Cancelled or not, the dialog is a plain data picker the next time.
    document.getElementById("dlg-path")
      .addEventListener("close", () => this.setPathDialogMode("data"));
  },

  /* The path browser does three jobs: picking data to import, picking a
   * project to open, and picking the folder projects are kept in. The mode
   * decides the title, the button and where "Open" sends what was picked. */
  setPathDialogMode(mode) {
    this.state.projectOpen = mode === "project";
    this.state.folderPick = mode === "folder";
    document.getElementById("dlg-path-title").textContent =
      mode === "project" ? "Open a project"
        : mode === "folder" ? "Choose the projects folder" : "Open by path";
    document.getElementById("btn-path-open").textContent =
      mode === "folder" ? "Use this folder" : "Open";
  },

  /* ".agrosuite" is what the file browser and the drop zone go by: the
   * server refuses to import one as a layer anyway, and the refusal would
   * only arrive after the upload. */
  isProjectFile(name) { return /\.agrosuite$/i.test(name || ""); },

  when(iso) {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? String(iso)
      : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  },

  /* ---------------------------------------------------------------- save */

  openSaveDialog() {
    if (!this.state.datasets.length) {
      this.toast("Nothing to save yet",
        "The session holds no datasets. Open a file first.", "warn");
      return;
    }
    const current = this.state.projectFile;
    document.getElementById("save-name").value = this.state.project?.name || "";
    // Saved to disk before: propose where it went, so re-saving is a matter
    // of pressing Save. A file named after the project keeps following the
    // name — the folder is proposed, and a renamed project gets a new file
    // beside the old one — while a file saved under a name of its own is
    // re-saved as it is. A temporary folder is never proposed.
    document.getElementById("save-path").value = !current?.on_disk ? ""
      : current.named_after_project ? current.folder : current.path;
    document.getElementById("dlg-save-project").showModal();
    document.getElementById("save-name").select();
  },

  async saveProject() {
    // busy() blocks the pointer, not the keyboard: a second Enter while the
    // first save is on its way would run the whole flow twice and race the
    // server's "already exists" check.
    if (this.state.saving) return;
    const dialog = document.getElementById("dlg-save-project");
    const name = (this.value("save-name") || "").trim();
    const path = (this.value("save-path") || "").trim();
    if (!name) {
      this.toast("Name the project", "The file is named after it.", "warn");
      return;
    }

    // The name given here is the project's: it names the file, it is what
    // the next save proposes, and it comes back when the file is reopened.
    // It travels with the save, so a refused save renames nothing.
    // Re-saving the file that is open is the normal flow and asks nothing —
    // the server knows which file that is; any other existing file is
    // replaced only once the person says so.
    const request = { name, path: path || null, overwrite: false };

    this.state.saving = true;
    let result;
    try {
      result = await this.busy(dialog, async () => {
        try {
          return await this.api("/api/session/save", { method: "POST", body: request });
        } catch (err) {
          if (err.status !== 409) throw err;
          if (!window.confirm(`${err.message}\n\nReplace it with the current session?`)) {
            return null;
          }
          return this.api("/api/session/save", {
            method: "POST", body: { ...request, overwrite: true },
          });
        }
      });
    } finally {
      this.state.saving = false;
    }
    if (!result) return;

    dialog.close();
    this.state.projectFile = result.file;
    this.renderProjectFile();
    this.refreshRecent();
    this.refreshProject();
    this.toast("Project saved",
      `${result.path}\n${result.datasets} dataset(s), ${this.fileSize(result.size_bytes)}.` +
      (result.file.on_disk ? "" : "\nThat folder goes when the app closes: keep the file " +
        "through the download link in the Load data panel.") +
      (result.warning ? `\n${result.warning}` : ""),
      result.warning ? "warn" : "");
  },

  /* What is open, where it is, and the way to keep it. */
  renderProjectFile() {
    const box = document.getElementById("project-file");
    const file = this.state.projectFile;
    if (!file) { box.innerHTML = ""; return; }
    box.innerHTML = `
      <div class="note ok" style="margin-top:8px">
        <b>${this.escape(file.name)}.agrosuite</b>${
          file.saved_at ? ` · saved ${this.escape(this.when(file.saved_at))}` : ""}
        ${file.on_disk ? `<br><span style="font:11px var(--mono);word-break:break-all">${
          this.escape(file.path)}</span>` : ""}
        ${file.download_url ? `<br><a href="${file.download_url}" download
          id="project-download">Download a copy</a>` : ""}
      </div>`;
  },

  /* -------------------------------------------------------------- recent */

  async refreshRecent() {
    const payload = await this.api("/api/session/recent").catch(() => null);
    if (!payload) return;
    this.state.recent = payload.recent;
    // The server remembers which file the session came from or last went
    // to; a copy kept only here would not survive a page reload, and the
    // next save would quietly go to the temporary folder.
    this.state.projectFile = payload.current;
    // And where the reader was, which is the other half of coming back.
    this.state.view = payload.view || this.state.view;
    this.renderProjectFile();
    this.renderRecentProjects();
    // Handed over once, by the server: work that appeared on screen without
    // being asked for has to be named, and the offer to undo it belongs
    // with the notice rather than in a menu.
    if (payload.resumed) this.announceResume(payload.resumed);
  },

  renderRecentProjects() {
    const box = document.getElementById("recent-projects");
    const items = this.state.recent || [];
    box.innerHTML = `<h4 style="margin:12px 0 4px">Recent projects</h4>`;
    if (!items.length) {
      box.innerHTML += '<p class="hint tight">None yet — "Save project" writes one.</p>';
      return;
    }
    const list = document.createElement("div");
    list.className = "file-list";
    for (const item of items) {
      const node = document.createElement("div");
      node.className = "item" + (item.exists ? "" : " unavailable") +
        (item.path === this.state.projectFile?.path ? " current" : "");
      node.title = item.exists ? item.path : `Not found: ${item.path}`;
      node.innerHTML = `<span class="icon">${item.exists ? "🗂" : "∅"}</span>
        <span class="name">${this.escape(item.name)}${item.saved_at
          ? `<span class="when">${this.escape(this.when(item.saved_at))}</span>` : ""}</span>`;
      node.addEventListener("click", () => {
        if (item.exists) { this.openProject(item.path); return; }
        this.toast("File not found", `${item.path}\nPlug the drive back in, or open ` +
          "the project by path from wherever it is now.", "warn");
      });
      list.appendChild(node);
    }
    box.appendChild(list);
  },

  /* ---------------------------------------------------------------- open */

  /* Opening and "New project" replace the session, so the person is asked
   * first. The client asks because the server cannot tell whether what it
   * holds was saved a minute ago or never. */
  confirmReplace(what) {
    const count = this.state.datasets.length;
    if (!count) return true;
    return window.confirm(`${what} closes the ${count} dataset(s) loaded now. ` +
      "Anything not saved to a project file is lost. Continue?");
  },

  /* A look at the file before anything is given up for it: what it holds
   * and when it was saved, or why it cannot be opened. Without it a folder,
   * or a ZIP of monitor data, went through "closes the N datasets" and only
   * then failed — the work was still there, but the question should never
   * have been asked. The server answers a refusal in words, not with an
   * error status: it is an answer, shown here. Null when refused. */
  async peekProject(path) {
    const peek = await this.api(`/api/session/peek?path=${encodeURIComponent(path)}`)
      .catch((err) => ({ ok: false, reason: err.message }));
    if (peek.ok) return peek;
    this.toast("Not a project to open", peek.reason, "warn");
    return null;
  },

  async openProject(path, peek = null) {
    peek = peek || await this.peekProject(path);
    if (!peek) return;
    const saved = peek.saved_at ? `, saved ${this.when(peek.saved_at)}` : "";
    if (!this.confirmReplace(
      `Opening '${peek.project || peek.name}' (${peek.datasets} dataset(s)${saved})`)) return;
    const result = await this.busy(document.querySelector("aside.left"), () =>
      this.api("/api/session/open", { method: "POST", body: { path } }));
    if (!result) return;
    await this.sessionReplaced(result);
  },

  async uploadProject(file) {
    if (!this.confirmReplace(`Opening '${file.name}'`)) return;
    const form = new FormData();
    form.append("file", file);
    const result = await this.busy(document.querySelector("aside.left"), () =>
      this.api("/api/session/upload", { method: "POST", body: form }));
    if (!result) return;
    // The copy lives in the session's temporary folder, not where the file
    // came from; the server reports it as not "on disk", and the save
    // dialog does not propose that folder.
    await this.sessionReplaced(result);
  },

  /* ``ask`` is false only where the question has already been answered:
   * the start-up toast, whose "Start a new project instead" closes a project
   * that was reopened from a file and is whole on disk. Asking "anything not
   * saved is lost" there would be asking about nothing. */
  async newProject(ask = true) {
    if (ask && !this.confirmReplace("A new project")) return;
    const ok = await this.busy(document.querySelector("aside.left"), () =>
      this.api("/api/session/new", { method: "POST" }));
    if (!ok) return;
    this.resetSessionState();
    this.state.projectFile = null;
    this.renderProjectFile();
    this.renderRecentProjects();
    await this.refreshDatasets();
    await this.refreshProject();
    this.renderTab();
    // What was open a moment ago is still on disk, whole. Naming it here is
    // the way back if "New project" — or the start-up toast's offer — was a
    // slip, without going looking for the file.
    // The status line belongs to the project that was open; asked for now
    // rather than at the next tick, it says "nothing to save yet" straight
    // away instead of naming the file for another three seconds.
    this.refreshAutosaveStatus();
    const latest = await this.api("/api/session/latest").catch(() => null);
    const last = latest?.project;
    this.toast("New project", "The session is empty. Load a file, or open a saved project.",
      "", last ? { label: `Reopen '${last.project}'`, run: () => this.openProject(last.path) }
               : null);
  },

  /* Everything the client cached about the old session goes: the report
   * caches keyed by dataset id, the map, the comparison. A reopened project
   * keeps its ids, so a stale cache would show a cleaning run after the
   * file was saved as if it were in the file. */
  resetSessionState() {
    if (this.state.compare) this.exitCompare({ reload: false });
    this.state.reports = {};
    this.state.joinReport = null;
    this.state.design = null;
    this.state.drawing = null;
    this.state.selectedId = null;
    this.state.selected = null;
    this.leaveTerrainMap();
    this.state.terrain = null;
    this.state.terrainRefusal = {};
    MapView.clearPoints();
    MapView.clearOverlays();
    this.clearRemovedOverlay();
    document.getElementById("legend").hidden = true;
    document.getElementById("map-status").textContent = "";
    // The skipped-files fold belongs to a drop made into the old session.
    this.renderDropReport([]);
  },

  async sessionReplaced(result) {
    this.resetSessionState();
    this.state.projectFile = result.file;
    // The trial layout is in the file with everything else: the Trial
    // design tab puts it back on the map when it renders, and Export
    // offers it as the prescription again.
    this.state.design = result.design?.result || null;
    this.renderProjectFile();
    await this.refreshDatasets();
    await this.refreshProject();
    this.refreshRecent();
    // Where the file says the work was left. A file saved before the view
    // was recorded says nothing, and the first dataset is picked as it
    // always was.
    this.state.view = result.view || { dataset_id: null, tab: null };
    // The tab is read out before anything is selected: selecting notes the
    // view, and what it would note is the tab the page is on now.
    const openOn = this.state.view.tab;
    const wanted = this.state.datasets.find((d) => d.id === this.state.view.dataset_id);
    const target = wanted || this.state.datasets[0];
    if (target) await this.selectDataset(target.id);
    else this.renderTab();
    this.openTab(openOn);
    this.refreshAutosaveStatus();
    this.toast(`Opened '${result.project}'`,
      `${result.datasets.length} dataset(s) from ${result.name}.agrosuite` +
      (result.saved_at ? `, saved ${this.when(result.saved_at)}.` : ".") +
      (result.warning ? `\n${result.warning}` : ""),
      result.warning ? "warn" : "");
  },

  /* ------------------------------------------------------ stored reports */

  /* The server keeps every report; the client caches only the ones it saw
   * being made. After a reopen the cache is empty while the reports are
   * there, so a tab asks for what it is missing instead of showing nothing. */
  async storedReport(kind) {
    const id = this.state.selectedId;
    const report = await this.api(`/api/datasets/${id}/report/${kind}`).catch(() => null);
    // The selection may have moved on while the request was out.
    if (!report || this.state.selectedId !== id) return null;
    return report;
  },

  /* POST /clean answers with the report plus the two datasets it produced;
   * the stored report is the report alone, so the ids are read off the
   * dataset list: the clean copy and the removed records both hang off the
   * dataset that was cleaned, and the last of each is the latest run. */
  cleanResultFor(id, report) {
    const children = this.state.datasets.filter((d) => d.parent_id === id);
    const clean = children.filter((d) => d.origin === "clean" && d.has_clean_report).pop();
    const removed = children.filter((d) => d.origin === "clean_removed").pop();
    return {
      report,
      clean: clean ? { id: clean.id } : null,
      removed: removed ? { id: removed.id } : null,
    };
  },

  async restoreCleanReport(d) {
    const shape = this.cleanResultFor(d.id, null);
    // A clean copy carries the report of its own making; it is shown on the
    // dataset that was cleaned, where "view clean data" leads somewhere.
    if (d.origin === "clean" && !shape.clean) return;
    const report = await this.storedReport("clean");
    if (!report || this.state.tab !== "limpeza") return;
    const result = { ...shape, report };
    this.state.reports[`${d.id}:clean`] = result;
    this.renderCleanReport(result);
  },

  async restoreDifmReport(d) {
    const report = await this.storedReport("difm");
    if (!report || this.state.tab !== "difm") return;
    this.state.reports[`${d.id}:difm`] = report;
    this.renderDifmReport(report);
  },
});

/* ======================================================================
 * Auto-save, the projects folder, and where the reader was
 *
 * The server writes the session to the projects folder as it is worked on
 * and reopens it at start-up; the page's part is three small things. It
 * says where that folder is and lets it be changed. It reports how the last
 * write went in one line — background work, reported the way background
 * work should be, with no dialog and nothing to dismiss. And it tells the
 * server which dataset is selected and which tab is open, because that is
 * the difference between reopening a project and coming back to it.
 * ==================================================================== */

Object.assign(App, {
  bindAutosave() {
    document.getElementById("autosave-toggle").addEventListener("change", (event) =>
      this.setAutosave(event.target.checked));
    document.getElementById("btn-projects-folder")
      .addEventListener("click", () => this.chooseProjectsFolder());
  },

  /* ------------------------------------------------------------ settings */

  async refreshSettings() {
    const payload = await this.api("/api/settings").catch(() => null);
    if (!payload) return;
    this.state.settings = payload;
    this.renderAutosave(payload.autosave_status);
    // A settings file that could not be read, or a folder that could not be
    // made, is said once when the app opens: the app works either way, on
    // the defaults, and silence would make the folder in the panel a lie.
    for (const problem of [payload.warning, payload.problem]) {
      if (problem) this.toast("Projects folder", problem, "warn");
    }
  },

  async setAutosave(on) {
    const result = await this.busy(document.getElementById("autosave"), () =>
      this.api("/api/settings", { method: "PUT", body: { autosave: on } }));
    if (!result) {
      // The switch shows what the server holds, not what was clicked.
      this.renderAutosave(this.state.settings?.autosave_status || {});
      return;
    }
    this.state.settings = result;
    this.renderAutosave(result.autosave_status);
    this.toast(on ? "Saving automatically" : "Auto-save off",
      on ? `This project is written to ${result.projects_dir} as you work.`
         : "Nothing is written from now on. 'Save project' still keeps a copy " +
           "wherever you point it, and what is already saved stays where it is.");
  },

  async chooseProjectsFolder() {
    // Asked again rather than remembered: the folder may have been made by
    // a save since the page loaded, and the browser cannot list one that is
    // not there yet — a folder waiting for its first save opens on home.
    await this.refreshSettings();
    this.setPathDialogMode("folder");
    const folder = this.state.settings?.projects_dir || "";
    document.getElementById("path-input").value = folder;
    document.getElementById("dlg-path").showModal();
    this.browse(this.state.settings?.exists ? folder : "");
    this.toast("Choose the projects folder",
      "Click through to the folder — or type its full path — and press " +
      "'Use this folder'. The project open now moves there with it.");
  },

  async applyProjectsFolder(path) {
    const result = await this.busy(document.getElementById("dlg-path"), () =>
      this.api("/api/settings", { method: "PUT", body: { projects_dir: path } }));
    if (!result) return;
    document.getElementById("dlg-path").close();
    this.state.settings = result;
    this.renderAutosave(result.autosave_status);
    this.toast("Projects folder", `${result.projects_dir}` +
      (result.moved ? `\n${result.to.split(/[\\/]/).pop()} moved there with it.` : "") +
      (result.problem ? `\n${result.problem}` : ""), result.problem ? "warn" : "");
  },

  /* -------------------------------------------------------- the one line */

  /* The status is read on a slow timer rather than after each action: the
   * write happens a couple of seconds after the change that caused it, so a
   * line refreshed on the click would always report the save before last.
   * Three seconds is cheap — one small JSON from a server on this machine —
   * and keeps "Saving…" visible for as long as it is true. */
  watchAutosave() {
    this.refreshAutosaveStatus();
    setInterval(() => this.refreshAutosaveStatus(), 3000);
  },

  async refreshAutosaveStatus() {
    const status = await this.api("/api/session/autosave").catch(() => null);
    if (status) this.renderAutosave(status);
  },

  renderAutosave(status) {
    document.getElementById("autosave-toggle").checked = !!status.enabled;
    document.getElementById("autosave-folder").textContent =
      status.folder || this.state.settings?.projects_dir || "";
    const line = document.getElementById("autosave-status");
    line.textContent = this.autosaveLine(status);
    line.classList.toggle("problem", status.state === "failed");

    // A failure is reported once, with the reason and what to do; the line
    // keeps it after the toast has gone. Repeating it every three seconds
    // would be the app shouting about a disk the person is already dealing
    // with, and saying nothing would hide it behind a small grey line.
    // The block above names "the file the session is on", and the server
    // fills that in the first time it writes one by itself. The page only
    // learns it by asking, so it asks once per new file rather than on
    // every tick of the status.
    if (status.path && status.path !== this.state.autosaveSeen) {
      this.state.autosaveSeen = status.path;
      this.refreshRecent();
    }

    if (status.state === "failed" && status.message !== this.state.autosaveProblem) {
      this.state.autosaveProblem = status.message;
      this.toast("Not saved automatically", status.message, "warn");
    }
    if (status.state !== "failed") this.state.autosaveProblem = null;
  },

  autosaveLine(status) {
    if (!status.enabled) return "Off — keep this session with 'Save project'.";
    if (status.state === "failed") return status.message;
    // Open twice: the other window has that project and keeps writing it,
    // so nothing here goes into it. Said in the line as well as in the
    // start-up toast, because the toast goes and the question — "why does
    // it not say Saved?" — comes back an hour later.
    if (status.state === "locked") return status.message;
    if (status.state === "saving") return "Saving…";
    if (status.saved_at) return `Saved ${this.clock(status.saved_at)}`;
    if (status.path) return `In ${status.path.split(/[\\/]/).pop()} — saved on the next change.`;
    return this.state.datasets.length ? "Saving shortly…" : "Nothing to save yet.";
  },

  clock(iso) {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? String(iso)
      : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  },

  /* ------------------------------------------------------ the start-up toast */

  /* The app reopened the last project by itself. That has to be said: work
   * appearing on screen nobody asked for is unsettling until it is named,
   * and the way out is one click here rather than a question asked at every
   * start-up — which is the same question, answered the same way, every day. */
  announceResume(resumed) {
    if (resumed.locked) {
      // Not a failure: the project is safe, in the window that has it. This
      // one is empty and ready to work — under another name, which is the
      // offer the toast carries.
      this.toast(`'${resumed.project}' is open in another window`, resumed.reason, "warn",
        { label: "Start a new project here", run: () => this.newProject(false) });
      return;
    }
    if (!resumed.ok) {
      this.toast("Could not pick up where you left off", resumed.reason, "warn");
      return;
    }
    // A newer file in the folder that would not open is named here rather
    // than left for him to find: the project that came back is the right one
    // to offer, but the one it walked past is the one he will look for.
    const skipped = (resumed.skipped || []).map((item) => `\n${item.name}: ${item.reason}`);
    this.toast(`Picked up '${resumed.project}'`,
      `${resumed.datasets} dataset(s) from ${resumed.name}.agrosuite` +
      (resumed.saved_at ? `, saved ${this.when(resumed.saved_at)}.` : ".") +
      (skipped.length ? `\nNewer, and could not be read:${skipped.join("")}` : ""),
      skipped.length ? "warn" : "",
      { label: "Start a new project instead", run: () => this.newProject(false) });
  },

  /* ------------------------------------------------------------ the view */

  /* Open a tab: from a click, or from where a project was left. */
  openTab(name) {
    if (!name || name === this.state.tab) return;
    const button = document.querySelector(`#steps button[data-tab="${name}"]`);
    // A tab named by a file this version no longer has is not an error to
    // report; the tab that is open stays open.
    if (!button) return;
    this.state.tab = name;
    for (const b of document.querySelectorAll("#steps button")) {
      b.setAttribute("aria-selected", String(b === button));
    }
    this.renderTab();
    this.noteView();
  },

  /* Tell the server where the reader is, so the project file can bring them
   * back to it. Only when it has actually changed — clicking the tab that is
   * already open is not a change worth writing a file for — and after a
   * pause, so that clicking through four tabs is one request and one save
   * rather than four of each. */
  noteView() {
    const view = { dataset_id: this.state.selectedId, tab: this.state.tab };
    const known = this.state.view || {};
    if (view.dataset_id === known.dataset_id && view.tab === known.tab) return;
    this.state.view = view;
    clearTimeout(this.state.viewTimer);
    this.state.viewTimer = setTimeout(() => {
      // Nothing on screen depends on this, and the next move sends it
      // again: a failure here is not worth a message.
      this.api("/api/session/view", { method: "PUT", body: view }).catch(() => {});
    }, 600);
  },

  /* Where the session says the reader was, once the page has the datasets:
   * after a reload, and after the app reopened a project at start-up. */
  async restoreView() {
    const wanted = this.state.datasets.find((d) => d.id === this.state.view?.dataset_id);
    const target = wanted || this.state.datasets[0];
    // Read out first, for the same reason as in sessionReplaced: selecting a
    // dataset notes the view, and by then this.state.view would say the tab
    // the page opened on rather than the one the session was left on.
    const openOn = this.state.view?.tab;
    // A full list over an empty map reads as if the data were gone, so
    // something is always selected when there is something to select.
    if (target && !this.state.selectedId) await this.selectDataset(target.id);
    this.openTab(openOn);
  },
});

/* ======================================================================
 * Printed report
 * ==================================================================== */

Object.assign(App, {
  /* A panel heading with the print button beside it. The button lives on
   * the report it prints, so what goes on paper is what is on screen. */
  printableHead(title, buttonId) {
    return `<div class="head"><h3>${title}</h3>
      <button class="small" id="${buttonId}"
              title="Write this report as a PDF, in the units shown on screen">
        Print report (PDF)</button></div>`;
  },

  /* The report belongs to the dataset that carries it: the clean copy after a
   * cleaning, the analysed dataset after the economics. The unit set travels with the
   * request because the server keeps no display preference — the numbers on
   * paper must be the ones the user was looking at, not the metric ones
   * underneath. The project name is the heading. */
  async printReport(datasetId, button) {
    const prefs = Units.get();
    const body = {
      dataset_id: datasetId,
      units: Object.fromEntries(["yield_unit", "input_rate_unit", "area_unit",
        "length_unit", "speed_unit", "currency", "crop"].map((k) => [k, prefs[k]])),
      title: this.state.project?.name || null,
    };
    const result = await this.busy(button, async () => {
      try {
        return await this.api("/api/report", { method: "POST", body });
      } catch (err) {
        if (err.status !== 404) throw err;
        // The dataset the button named is no longer in the session: removed
        // from elsewhere — the MCP server, another tab — since this panel was
        // drawn. The list is the truth: refreshing it prunes the cached
        // result that carried the old id, and redrawing the tab rebuilds the
        // panel from what is still loaded. For a cleaning report that is the
        // original, which keeps its own copy and prints from that; when the
        // selected dataset itself is gone there is nothing to redraw for.
        const wasSelected = datasetId === this.state.selectedId;
        if (wasSelected) this.clearSelection();
        await this.refreshDatasets();
        this.renderTab();
        this.toast(wasSelected ? "That dataset is gone" : "The clean copy is gone",
          "It was removed from the session after this panel was drawn. The list " +
          "has been refreshed" + (wasSelected ? "; pick a dataset to go on."
            : " and the report redrawn from the original's own copy; press Print again."),
          "warn");
        return null;
      }
    });
    if (!result) return;

    // The file is served as an attachment, so the new tab is the download.
    // A pop-up blocker returns null; a clicked anchor needs no permission,
    // and the toast names the path on disk either way.
    if (!window.open(result.download_url, "_blank")) {
      const link = document.createElement("a");
      link.href = result.download_url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    }

    // The stored report keys are the wire names; these are what they are
    // called on screen and on the page.
    const names = { preflight: "first look", clean: "cleaning", difm: "economic report",
                    terrain: "relief" };
    const printed = result.sections.map((s) => names[s]).filter(Boolean).join(", ");
    this.toast("Report written",
      `${result.filename}: ${printed || "header only"}, ` +
      `${result.pages} page${result.pages === 1 ? "" : "s"}.\nSaved at ${result.path}`);
  },
});
