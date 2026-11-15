"use strict";

const byId = id => document.getElementById(id);
const data = JSON.parse(byId("report-data").textContent);
const canvas = byId("scene");
const context = canvas.getContext("2d");
const chart = byId("section-chart");
const scatter = byId("scatter-chart");
const initialView = {yaw: -.43, pitch: .58, zoom: 1};
const view = {...initialView};
const chosen = {case1: null, case2: null};
const formatter = (digits = 0) => new Intl.NumberFormat("ru-RU", {
  minimumFractionDigits: digits, maximumFractionDigits: digits
});
const formats = [0, 1, 2, 3].map(formatter);
const fmt = (value, digits = 0) => Number.isFinite(value) ? formats[digits].format(value) : "—";
const mm = value => Number.isFinite(value) ? fmt(value, 2) + " мм" : "—";
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const setText = (id, text) => { byId(id).textContent = text; };
const hypot = Math.hypot;
const predictions = {
  case1: "GAM + CatBoost",
  case2: "Random Forest"
};
let activeCase = "case1";
let rows, route, bounds, priorityRows, domain;
let screenPoints = [];
let frame = 0;

function colour(ratio, light = 1) {
  const value = clamp((ratio - .45) / .55, 0, 1) * 2;
  const stops = [[231, 135, 102], [219, 190, 114], [91, 214, 180]];
  const index = Math.min(1, Math.floor(value));
  return "rgb(" + stops[index].map((base, i) =>
    Math.round(clamp((base + (stops[index + 1][i] - base) * (value - index)) * light, 0, 255))
  ).join(",") + ")";
}

function routeBounds(points) {
  const usable = points.filter(Boolean);
  if (!usable.length) return null;
  const min = [0, 1, 2].map(i => Math.min(...usable.map(point => point[i])));
  const max = [0, 1, 2].map(i => Math.max(...usable.map(point => point[i])));
  return {min, max, mid: min.map((value, i) => (value + max[i]) / 2), count: usable.length};
}

function drawScene() {
  frame = 0;
  if (!context) return;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (width < 1 || height < 1) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!bounds) {
    context.textAlign = "center";
    context.fillStyle = "#a9bdc0";
    context.font = "14px system-ui";
    context.fillText("Координаты трассы не заданы", width / 2, height / 2);
    return;
  }

  const heightScale = Number(byId("height-scale").value);
  const [east, north, elevation] = bounds.mid;
  const span = Math.max(1, hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1],
    (bounds.max[2] - bounds.min[2]) * heightScale));
  const scale = Math.min(width * .80, height * 1.15) / span * view.zoom;
  const sy = Math.sin(view.yaw), cy = Math.cos(view.yaw);
  const sp = Math.sin(view.pitch), cp = Math.cos(view.pitch);
  function project(x, z, h) {
    const dx = x - east, dz = z - north, dy = (h - elevation) * heightScale;
    const forward = dx * sy + dz * cy;
    const depth = forward * cp + dy * sp;
    return [width / 2 + scale * (dx * cy - dz * sy), height * .49 - scale * (dy * cp - forward * sp), depth];
  }

  const gridY = bounds.min[2] - span * .055 / heightScale;
  const gridSpan = span * .57;
  context.lineWidth = 1;
  context.strokeStyle = "#294651";
  context.beginPath();
  for (let index = -5; index <= 5; index++) {
    const offset = gridSpan * index / 5;
    for (const [a, b] of [
      [project(east - gridSpan, north + offset, gridY), project(east + gridSpan, north + offset, gridY)],
      [project(east + offset, north - gridSpan, gridY), project(east + offset, north + gridSpan, gridY)]
    ]) {
      context.moveTo(a[0], a[1]);
      context.lineTo(b[0], b[1]);
    }
  }
  context.stroke();

  screenPoints = route.map(point => point && project(...point));
  const radius = clamp(Math.min(width, height) * .028 * Math.sqrt(view.zoom), 4, 21);
  const slices = [];
  // Draw each span from far to near. Concentric screen strokes shade the round cross-section.
  for (let i = 1; i < screenPoints.length; i++) {
    const a = screenPoints[i - 1], b = screenPoints[i];
    if (a && b) slices.push({a, b, depth: (a[2] + b[2]) / 2, i});
  }
  slices.sort((a, b) => a.depth - b.depth);
  context.lineCap = "round";
  for (const {a, b, i} of slices) {
    if (Math.max(a[0], b[0]) < -radius || Math.min(a[0], b[0]) > width + radius ||
        Math.max(a[1], b[1]) < -radius || Math.min(a[1], b[1]) > height + radius) continue;
    const ratio = rows[i].thickness_pred / rows[i].nominal_mm;
    for (const [weight, shift, light] of [[2.1, 0, .36], [1.85, -.09, .67], [1.39, -.22, .93],
      [.77, -.37, 1.12], [.21, -.55, 1.28]]) {
      context.lineWidth = radius * weight;
      context.strokeStyle = colour(ratio, light);
      context.beginPath();
      context.moveTo(a[0], a[1] + radius * shift);
      context.lineTo(b[0], b[1] + radius * shift);
      context.stroke();
    }
  }
  const occupied = new Set();
  const pointMarker = (point, size, color, fill) => {
    context.beginPath();
    context.arc(point[0], point[1], size, 0, 2 * Math.PI);
    context.strokeStyle = color;
    context.lineWidth = 1.7;
    if (fill) {
      context.fillStyle = fill;
      context.fill();
    }
    context.stroke();
  };
  for (const row of priorityRows) {
    const point = screenPoints[row.segment_id];
    if (!point) continue;
    const bucket = Math.round(point[0] / 13) + "/" + Math.round(point[1] / 13);
    if (occupied.has(bucket)) continue;
    occupied.add(bucket);
    pointMarker(point, 2.2, "#ffd28c", "#33291e");
  }
  const selected = screenPoints[chosen[activeCase]];
  if (selected) {
    pointMarker(selected, radius + 4, "#f7fff9");
    pointMarker(selected, 4, "#f7fff9", "#15302e");
  }

  // A north arrow uses the same projection as the route.
  const zero = project(east, north, gridY);
  const northPoint = project(east, north + span * .1, gridY);
  const bearing = Math.atan2(northPoint[1] - zero[1], northPoint[0] - zero[0]);
  context.save();
  context.translate(width - 31, height - 32);
  context.rotate(bearing);
  context.fillStyle = "#a9bdc0";
  context.beginPath();
  context.moveTo(14, 0);
  context.lineTo(-6, -5);
  context.lineTo(-2, 0);
  context.lineTo(-6, 5);
  context.closePath();
  context.fill();
  context.restore();
  context.fillStyle = "#a9bdc0";
  context.font = "11px system-ui";
  context.textAlign = "center";
  context.fillText("С", width - 31, height - 10);
}

function redrawScene() {
  if (!frame) frame = requestAnimationFrame(drawScene);
}

function svgElement(svg, tag, attrs, label) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  if (label !== undefined) element.textContent = label;
  svg.append(element);
}

function axes(svg, minX, maxX, minY, maxY, xTitle, yTitle, formatX, formatY) {
  const width = svg.clientWidth, height = svg.clientHeight;
  const left = 44, top = 24, right = width - 17, bottom = height - 42;
  const x = value => left + (value - minX) / (maxX - minX) * (right - left);
  const y = value => bottom - (value - minY) / (maxY - minY) * (bottom - top);
  const text = {fill: "#a9bdc0", "font-size": "11"};
  svg.setAttribute("viewBox", [0, 0, width, height].join(" "));
  svg.replaceChildren();
  const count = width < 500 ? 3 : 5;
  for (let i = 0; i <= count; i++) {
    const px = x(minX + (maxX - minX) * i / count);
    svgElement(svg, "line", {x1: px, y1: top, x2: px, y2: bottom, stroke: "#23383e"});
    svgElement(svg, "text", {...text, x: px, y: bottom + 16, "text-anchor": "middle"}, formatX(i / count));
  }
  for (let i = 0; i <= 4; i++) {
    const py = y(minY + (maxY - minY) * i / 4);
    svgElement(svg, "line", {x1: left, y1: py, x2: right, y2: py, stroke: "#31474d"});
    svgElement(svg, "text", {...text, x: left - 7, y: py + 4, "text-anchor": "end"}, formatY(i / 4));
  }
  svgElement(svg, "text", {...text, x: left, y: 13}, yTitle);
  svgElement(svg, "text", {...text, x: (left + right) / 2, y: height - 6, "text-anchor": "middle"}, xTitle);
  return {x, y, left, right, top, bottom};
}

function renderProfile() {
  if (chart.clientWidth < 80) return;
  const selected = rows[chosen[activeCase]];
  const length = rows[rows.length - 1].end_m;
  const range = Math.min(Number(byId("profile-window").value) || length, length);
  const start = clamp(selected.center_m - range / 2, 0, length - range), end = start + range;
  const visible = rows.filter(row => row.center_m >= start && row.center_m <= end);
  const min = Math.max(0, Math.floor(Math.min(...visible.flatMap(row =>
    [row.thickness_pred, row.anchor_mm].filter(Number.isFinite))) - .2));
  const max = Math.ceil(Math.max(...visible.flatMap(row =>
    [row.thickness_pred, row.nominal_mm, row.anchor_mm].filter(Number.isFinite))) + .15);
  const {x, y, left, right, top, bottom} = axes(chart, start, end, min, max, "Расстояние, км", "Толщина, мм",
    fraction => fmt((start + range * fraction) / 1000, range < 1000 ? 2 : 1),
    fraction => fmt(min + (max - min) * fraction, 1));
  domain = {start, range, left, right, top, bottom};
  svgElement(chart, "line", {x1: left, y1: y(selected.nominal_mm), x2: right, y2: y(selected.nominal_mm),
    stroke: "#a2b5b8", "stroke-dasharray": "5 5"});
  for (const [column, stroke, opacity] of [["anchor_mm", "#92aafa", .76], ["thickness_pred", "#73dfc5", 1]]) {
    let connected = false;
    const commands = [];
    for (const row of visible) {
      if (Number.isFinite(row[column])) {
        const px = x(row.center_m).toFixed(1), py = y(row[column]).toFixed(1);
        commands.push((connected ? "L" : "M") + px + "," + py);
        if (column === "anchor_mm") commands.push("l.01,0");
        connected = true;
      } else {
        connected = false;
      }
    }
    svgElement(chart, "path", {d: commands.join(" "), stroke, fill: "none", opacity, "stroke-width": 1.5, "stroke-linecap": "round"});
  }
  for (const row of visible) {
    if (row.rank !== null) svgElement(chart, "circle", {cx: x(row.center_m), cy: y(row.thickness_pred),
      r: 2.5, stroke: "#ffd28c", fill: "#23322c"});
  }
  const px = x(selected.center_m);
  svgElement(chart, "line", {x1: px, y1: top, x2: px, y2: bottom, stroke: "#d4ece8", opacity: .75});
  svgElement(chart, "circle", {cx: px, cy: y(selected.thickness_pred), r: 4.5, fill: "#fff", stroke: "#143f34"});
}

function renderQuality() {
  if (scatter.clientWidth < 80) return;
  const warm = activeCase === "case1";
  const split = warm ? "late_2023" : "holdout_main";
  const rows = data.observations.filter(row => row.case === activeCase && row.split === split);
  const closed = data.summary.closed_validation?.[activeCase];
  const open = warm
    ? data.metrics.find(row => row.case === activeCase && row.model === "direct_blend" && row.split === split)
    : data.summary.case2_macro_validation.find(row => row.model === "xgboost_knn" && row.seed === data.summary.seed);
  setText("closed-pearson", fmt(closed?.pearson_squared, 3));
  setText("closed-mae", fmt(closed?.mae_mm, 3));
  setText("closed-note", closed ? "Оценка организаторов, сообщённая командой для этого result.csv." : "Для этого файла оценка не зарегистрирована.");
  setText("open-pearson", fmt(open?.pearson_squared, 3));
  setText("open-mae", fmt(open?.mae_mm, 3));
  setText("open-note", warm ? "Поздние ВТД соседних участков." : "Сначала среднее по кампаниям, затем по магистралям.");
  setText("scatter-meta", fmt(rows.length) + " меток · " + (warm ? "поздние ВТД" : "MAIN-2 и MAIN-3 по очереди вне обучения"));
  if (!rows.length) {
    scatter.replaceChildren();
    return;
  }
  const values = rows.flatMap(row => [row.y_mm, row.thickness_pred]);
  const min = Math.floor(Math.min(...values)), max = Math.ceil(Math.max(...values)) || 1;
  const range = Math.max(1, max - min);
  const tick = fraction => fmt(min + range * fraction, 1);
  const {x, y} = axes(scatter, min, min + range, min, min + range, "Факт, мм", "Прогноз, мм", tick, tick);
  svgElement(scatter, "line", {x1: x(min), y1: y(min), x2: x(min + range), y2: y(min + range), stroke: "#9aadb0",
    "stroke-dasharray": "5 5"});
  for (const row of rows) {
    svgElement(scatter, "circle", {cx: x(row.y_mm), cy: y(row.thickness_pred), r: warm ? 3.7 : 2.7,
      fill: "#73dfc5", opacity: warm ? .77 : .25});
  }
}

function selectSegment(index) {
  if (!Number.isFinite(index)) index = chosen[activeCase] ?? 0;
  index = clamp(Math.round(index), 0, rows.length - 1);
  chosen[activeCase] = index;
  const row = rows[index];
  byId("segment").value = index;
  byId("priority").value = row.rank === null ? "" : String(index);
  byId("prediction").replaceChildren(document.createTextNode(fmt(row.thickness_pred, 2) + " "));
  const unit = document.createElement("small");
  unit.textContent = "мм";
  byId("prediction").append(unit);
  setText("segment-band", fmt(row.start_m) + "–" + fmt(row.end_m, row.end_m % 1 ? 2 : 0) + " м от начала");
  setText("fraction", fmt(100 * row.thickness_pred / row.nominal_mm, 1) + "%");
  setText("nominal", mm(row.nominal_mm));
  setText("anchor", mm(row.anchor_mm));
  setText("altitude", route[index] ? fmt(route[index][2], 1) + " м" : "—");
  renderProfile();
  redrawScene();
}

function switchCase(next) {
  activeCase = next;
  rows = data.profiles.filter(item => item.case === next).sort((a, b) => a.segment_id - b.segment_id);
  route = data.routes[rows[0].id] ?? [];
  bounds = routeBounds(route);
  screenPoints = [];
  priorityRows = rows.filter(row => row.rank !== null).sort((a, b) => a.rank - b.rank);
  const priority = byId("priority");
  priority.replaceChildren(new Option("Выбрать · " + fmt(priorityRows.length) + " сегм.", ""));
  for (const row of priorityRows) priority.add(new Option(
    "№" + row.rank + " · сегм. " + row.segment_id + " · " + mm(row.thickness_pred), row.segment_id));
  setText("ranking-model", predictions[next] + " формирует список приоритетов.");
  const date = rows[0].target_date.slice(0, 10).split("-").reverse().join(".");
  setText("route-meta", rows[0].id + " / " + date + " / " + fmt(rows[rows.length - 1].end_m / 1000, 2) + " км");
  setText("route-coverage", bounds && bounds.count < rows.length ? fmt(bounds.count) + " из " + fmt(rows.length) + " сегм. с координатами" : "");
  byId("segment").max = rows.length - 1;
  byId("anchor-key").hidden = !rows.some(row => row.anchor_observed);
  for (const button of document.querySelectorAll("[data-case]")) {
    button.setAttribute("aria-pressed", String(button.dataset.case === next));
  }
  Object.assign(view, initialView);
  selectSegment(chosen[next] ?? priorityRows[0]?.segment_id ?? 0);
  renderQuality();
}

function zoom(factor) {
  view.zoom = clamp(view.zoom * factor, .45, 5);
  redrawScene();
}

byId("height-scale").addEventListener("change", redrawScene);
byId("profile-window").addEventListener("change", renderProfile);
byId("segment").addEventListener("change", event => selectSegment(event.target.valueAsNumber));
byId("priority").addEventListener("change", event => {
  if (event.target.value) selectSegment(Number(event.target.value));
});
byId("zoom-in").addEventListener("click", () => zoom(1.23));
byId("zoom-out").addEventListener("click", () => zoom(1 / 1.23));
byId("reset-view").addEventListener("click", () => {
  Object.assign(view, initialView);
  redrawScene();
});
canvas.addEventListener("keydown", event => {
  const key = event.key;
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"].includes(key)) return;
  event.preventDefault();
  if (key === "ArrowLeft") view.yaw -= .1;
  if (key === "ArrowRight") view.yaw += .1;
  if (key === "ArrowUp") view.pitch = clamp(view.pitch + .1, -.95, 1.4);
  if (key === "ArrowDown") view.pitch = clamp(view.pitch - .1, -.95, 1.4);
  if (key === "+" || key === "=") zoom(1.23);
  if (key === "-") zoom(1 / 1.23);
  if (key === "Home") Object.assign(view, initialView);
  redrawScene();
});
canvas.addEventListener("wheel", event => {
  if (document.activeElement !== canvas) return;
  event.preventDefault();
  zoom(event.deltaY < 0 ? 1.13 : 1 / 1.13);
}, {passive: false});
let drag;
canvas.addEventListener("pointerdown", event => {
  if (!event.isPrimary || event.button !== 0) return;
  drag = {x: event.clientX, y: event.clientY, moved: false, pointerId: event.pointerId};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (drag?.pointerId !== event.pointerId) return;
  const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
  if (!drag.moved && hypot(dx, dy) < 4) return;
  Object.assign(drag, {x: event.clientX, y: event.clientY, moved: true});
  view.yaw += dx * .009;
  view.pitch = clamp(view.pitch + dy * .009, -.95, 1.4);
  redrawScene();
});
canvas.addEventListener("pointerup", event => {
  if (drag?.pointerId !== event.pointerId) return;
  const moved = drag.moved;
  drag = null;
  if (moved) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let nearest = null, best = 24;
  screenPoints.forEach((point, i) => {
    if (!point) return;
    const d = hypot(point[0] - x, point[1] - y);
    if (d < best) {
      nearest = i;
      best = d;
    }
  });
  if (nearest !== null) selectSegment(nearest);
});
canvas.addEventListener("lostpointercapture", () => { drag = null; });
chart.addEventListener("click", event => {
  if (!domain) return;
  const rect = chart.getBoundingClientRect();
  const px = event.clientX - rect.left, py = event.clientY - rect.top;
  if (px < domain.left || px > domain.right || py < domain.top || py > domain.bottom) return;
  const position = domain.start + domain.range * (px - domain.left) / (domain.right - domain.left);
  const nearest = rows.reduce((best, row) =>
    Math.abs(row.center_m - position) < Math.abs(best.center_m - position) ? row : best);
  selectSegment(nearest.segment_id);
});
for (const button of document.querySelectorAll("[data-case]")) {
  button.addEventListener("click", () => switchCase(button.dataset.case));
}
for (const button of document.querySelectorAll("[data-file]")) {
  button.addEventListener("click", () => {
    const name = button.dataset.file;
    const type = name.endsWith(".json") ? "application/json" : "text/csv";
    const url = URL.createObjectURL(new Blob([data.downloads[name]], {type: type + ";charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
switchCase(activeCase);
const observer = new ResizeObserver(entries => {
  for (const {target} of entries) {
    if (target === canvas) redrawScene();
    if (target === chart) renderProfile();
    if (target === scatter) renderQuality();
  }
});
for (const element of [canvas, chart, scatter]) observer.observe(element);
