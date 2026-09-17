/* Unit conversion and formatting, on the client.
 *
 * The server always speaks internal units (kg/ha, ha, m, km/h, kg). Every
 * conversion to what the user picked happens here, at display time — so no
 * stored number depends on a screen preference, and switching from acres to
 * hectares requires no reload. */

const Units = (() => {
  let catalog = null;
  let prefs = null;

  /* Bushel test weight in kg, per crop. Comes from the server. */
  function bushelKg(cropKey) {
    const crop = (catalog?.crops || []).find((c) => c.key === cropKey);
    return crop ? crop.bushel_kg : 25.4012;
  }

  /* Factor taking the chosen unit to the internal one. Bushels are resolved
   * here because they depend on the crop, which the user can change at any time. */
  function factor(group, unitKey) {
    const info = catalog?.groups?.[group];
    if (!info) return 1;
    const entry = info.units.find((u) => u.key === unitKey);
    if (!entry) return 1;
    if (entry.factor != null) return entry.factor;
    const weight = bushelKg(prefs.crop);
    if (group === "mass") return weight;
    if (group === "rate_mass") return weight / catalog.bushel_kg_per_acre_divisor;
    return 1;
  }

  function init(unitCatalog, presetKey) {
    catalog = unitCatalog;
    applyPreset(presetKey || catalog.default_preset);
  }

  function applyPreset(key) {
    const preset = catalog.presets[key] || catalog.presets[catalog.default_preset];
    prefs = { preset: key, ...preset };
    return prefs;
  }

  function set(key, value) { prefs[key] = value; }
  function get() { return prefs; }
  function all() { return catalog; }

  function currencySymbol() {
    const entry = (catalog?.currencies || []).find((c) => c.key === prefs.currency);
    return entry ? entry.symbol : "$";
  }

  /* --- named conversions, one per quantity shown in the interface --- */

  const convert = {
    yield: (v) => (v == null ? null : v / factor("rate_mass", prefs.yield_unit)),
    inputRate: (v) => (v == null ? null : v / factor("rate_mass", prefs.input_rate_unit)),
    area: (v) => (v == null ? null : v / factor("area", prefs.area_unit)),
    length: (v) => (v == null ? null : v / factor("length", prefs.length_unit)),
    speed: (v) => (v == null ? null : v / factor("speed", prefs.speed_unit)),
    mass: (v) => (v == null ? null : v / factor("mass", prefs.mass_unit)),
  };

  /* The other direction: screen unit to internal, on the way to the server. */
  const toInternal = {
    yield: (v) => (v == null ? null : v * factor("rate_mass", prefs.yield_unit)),
    inputRate: (v) => (v == null ? null : v * factor("rate_mass", prefs.input_rate_unit)),
    area: (v) => (v == null ? null : v * factor("area", prefs.area_unit)),
    length: (v) => (v == null ? null : v * factor("length", prefs.length_unit)),
    speed: (v) => (v == null ? null : v * factor("speed", prefs.speed_unit)),
  };

  const label = {
    yield: () => prefs.yield_unit,
    inputRate: () => prefs.input_rate_unit,
    area: () => prefs.area_unit,
    length: () => prefs.length_unit,
    speed: () => prefs.speed_unit,
    mass: () => prefs.mass_unit,
  };

  /* Operations where the main variable is harvested production. In every other
   * one — application, seeding, prescription — it is an input rate, and showing
   * lb/ac of fertilizer as bu/ac of grain would be a gross error. */
  const YIELD_OPERATIONS = new Set(["harvest"]);

  /* Pick the right conversion from the canonical column name and the dataset's
   * operation type. */
  function forColumn(column, operation) {
    const isYield = operation == null
      ? true                              // with no context, yield is the default
      : YIELD_OPERATIONS.has(operation);
    switch (column) {
      case "value":
        return isYield
          ? { conv: convert.yield, unit: label.yield(), kind: "yield" }
          : { conv: convert.inputRate, unit: label.inputRate(), kind: "inputRate" };
      case "target_rate":
      case "applied_rate":
        return { conv: convert.inputRate, unit: label.inputRate(), kind: "inputRate" };
      case "speed_kmh":
        return { conv: convert.speed, unit: label.speed(), kind: "speed" };
      case "swath_m":
      case "distance_m":
      case "elev_m":
        return { conv: convert.length, unit: label.length(), kind: "length" };
      case "moisture_pct":
        return { conv: (v) => v, unit: "%", kind: null };
      default:
        return { conv: (v) => v, unit: "", kind: null };
    }
  }

  /* Number formatting with decimal places proportional to magnitude — 0.84
   * needs two places; 12,553 needs none. */
  function num(value, decimals) {
    if (value == null || !isFinite(value)) return "—";
    let places = decimals;
    if (places == null) {
      const abs = Math.abs(value);
      // An exact zero gets no decimals: "0.000" in a rate table suggests a
      // precision that is not there.
      places = abs === 0 ? 0 : abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 1 ? 2 : 3;
    }
    return value.toLocaleString("en-CA", {
      minimumFractionDigits: places,
      maximumFractionDigits: places,
    });
  }

  function money(value, decimals = 2) {
    if (value == null || !isFinite(value)) return "—";
    return `${currencySymbol()} ${num(value, decimals)}`;
  }

  /* Value with its unit, already converted from the internal one. */
  function show(value, kind, decimals) {
    const converted = convert[kind] ? convert[kind](value) : value;
    const unit = label[kind] ? label[kind]() : "";
    return `${num(converted, decimals)}${unit ? " " + unit : ""}`;
  }


  /* The mass (kg) and area (ha) behind a rate unit such as "bu/ac". A price is
   * always quoted per numerator — dollars per bushel, per pound — so converting
   * one means splitting the two halves apart. */
  const MASS_PART = { kg: 1, t: 1000, sc: 60, lb: 0.45359237, g: 0.001, bu: null };
  const AREA_PART = { ha: 1, ac: 0.40468564224, alq: 2.42, m2: 1e-4 };

  function rateUnitParts(unitKey) {
    const [num, den] = String(unitKey || "").split("/");
    let mass = MASS_PART[num];
    if (mass === null || mass === undefined) {
      mass = num === "bu" ? bushelKg(prefs.crop) : null;
    }
    return { massKg: mass, areaHa: AREA_PART[den] ?? null };
  }

  /* Price quoted per numerator unit -> price per kg (internal). */
  function priceToInternal(price, rateUnitKey) {
    const { massKg } = rateUnitParts(rateUnitKey);
    return massKg ? price / massKg : price;
  }

  function priceFromInternal(pricePerKg, rateUnitKey) {
    const { massKg } = rateUnitParts(rateUnitKey);
    return massKg ? pricePerKg * massKg : pricePerKg;
  }

  /* The numerator's label, for writing "C$ / bu" on the form. */
  function rateNumerator(unitKey) {
    return String(unitKey || "").split("/")[0] || "";
  }

  return {
    init, applyPreset, set, get, all, factor, forColumn,
    convert, toInternal, label, num, money, show, currencySymbol, bushelKg,
    rateUnitParts, priceToInternal, priceFromInternal, rateNumerator,
  };
})();
