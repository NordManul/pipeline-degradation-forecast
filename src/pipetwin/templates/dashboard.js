"use strict";

const byId = id => document.getElementById(id);
const data = JSON.parse(byId("report-data").textContent);
const canvas = byId("scene");
const context = canvas.getContext("2d");
const routePanel = byId("route-panel");
const fullscreenButton = byId("fullscreen-view");
const mapScale = byId("map-scale");
const chart = byId("section-chart");
const scatter = byId("scatter-chart");
const initialView = {yaw: -.43, pitch: .58, zoom: 1, target: null, offsetX: 0, offsetY: 0, follow: false};
const minPitch = .18;
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
const rankingModels = {
  case1: "GAM + CatBoost",
  case2: "Random Forest"
};
const forecastModels = {
  case1: "GAM + CatBoost",
  case2: "XGBoost + KNN"
};
const caseTabs = Array.from(document.querySelectorAll("[data-case]"));
const ringOffsets = Array.from({length: 10}, (_, i) => {
  const angle = i * 2 * Math.PI / 10;
  return [Math.cos(angle), Math.sin(angle)];
});
const lighting = [.58, .66, .74, .82, .90, .98, 1.06, 1.14];
let activeCase = "case1";
let rows, route, bounds, priorityRows, domain, minThickness, maxThickness;
let screenPoints = [];
let mesh;
let hitRadius = 24;
let frame = 0;

function colour(thickness, light = 1) {
  const span = maxThickness - minThickness;
  const value = span > 0 ? 2 * clamp((thickness - minThickness) / span, 0, 1) : 1;
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

function niceDistance(limit) {
  const power = 10 ** Math.floor(Math.log10(limit));
  const magnitude = limit / power;
  return power * (magnitude >= 5 ? 5 : magnitude >= 2 ? 2 : 1);
}

function distanceLabel(metres) {
  if (metres >= 1000) return fmt(metres / 1000, metres % 1000 ? 1 : 0) + " км";
  return fmt(metres, metres < 1 ? 1 : 0) + " м";
}

function drawGrid(project, unproject, step, width, height, eastward) {
  // Clip the work to the part of the horizontal plane visible on screen.
  const inset = 3;
  const corners = [[-inset, -inset], [width + inset, -inset],
    [width + inset, height + inset], [-inset, height + inset]].map(([x, y]) => unproject(x, y));
  const [left, bottom] = [0, 1].map(axis => Math.floor(Math.min(...corners.map(p => p[axis])) / step));
  const [right, top] = [0, 1].map(axis => Math.ceil(Math.max(...corners.map(p => p[axis])) / step));
  const line = (a, b) => {
    context.moveTo(a[0], a[1]);
    context.lineTo(b[0], b[1]);
  };
  context.lineWidth = 1;
  context.strokeStyle = "#294651";
  context.beginPath();
  for (let i = left; i <= right; i++) line(project(i * step, bottom * step), project(i * step, top * step));
  for (let i = bottom; i <= top; i++) line(project(left * step, i * step), project(right * step, i * step));
  context.stroke();

  // Mark one actual grid edge near the lower-left of the view.
  const targetX = Math.min(.32 * width, 136), targetY = height - 43;
  const [east, north] = unproject(targetX, targetY);
  const alongX = eastward ? 1 : 0, alongY = 1 - alongX;
  const startX = Math.round(east / step - alongX / 2);
  const startY = Math.round(north / step - alongY / 2);
  let edge = null, closest = Infinity;
  for (let i = -2; i <= 2; i++) {
    for (let j = -2; j <= 2; j++) {
      const ix = startX + i, iy = startY + j;
      const a = project(ix * step, iy * step), b = project((ix + alongX) * step, (iy + alongY) * step);
      if ([a, b].some(([x, y]) => x < 12 || x > width - 12 || y < 12 || y > height - 31)) continue;
      const x = (a[0] + b[0]) / 2, y = (a[1] + b[1]) / 2;
      const distance = (x - targetX) ** 2 + (y - targetY) ** 2;
      if (distance < closest) {
        closest = distance;
        edge = {a, b, x, y};
      }
    }
  }
  mapScale.hidden = !edge;
  if (!edge) return;
  const {a, b, x, y} = edge;
  const lengthPx = hypot(b[0] - a[0], b[1] - a[1]);
  const tx = 4 * (a[1] - b[1]) / lengthPx, ty = 4 * (b[0] - a[0]) / lengthPx;
  context.strokeStyle = "#9eb8c1";
  context.lineWidth = 1.6;
  context.beginPath();
  line(a, b);
  for (const p of [a, b]) line([p[0] - tx, p[1] - ty], [p[0] + tx, p[1] + ty]);
  context.stroke();
  const label = distanceLabel(step);
  if (mapScale.textContent !== label) mapScale.textContent = label;
  mapScale.style.left = Math.round(x) + "px";
  mapScale.style.top = Math.round(y + 8) + "px";
}

function prepareMesh(heightScale) {
  const points = route.map(point => point && [point[0], point[1], point[2] * heightScale]);
  const ringAt = index => {
    const a = points[index - 1], b = points[index];
    const centre = a && b ? a.map((v, i) => (v + b[i]) / 2) : a ?? b;
    if (!centre) return null;
    const from = points[Math.max(0, index - 3)] ?? a ?? centre;
    const to = points[Math.min(points.length - 1, index + 2)] ?? b ?? centre;
    const [dx, dz, dy] = to.map((v, i) => v - from[i]);
    const heading = Math.atan2(dz, dx);
    const incline = Math.atan2(dy, hypot(dx, dz));
    const sh = Math.sin(heading), ch = Math.cos(heading);
    const si = Math.sin(incline), ci = Math.cos(incline);
    const offsets = ringOffsets.map(([lateral, vertical]) =>
      [-lateral * sh - vertical * si * ch, lateral * ch - vertical * si * sh, vertical * ci]);
    return {centre, offsets};
  };
  const rings = Array.from({length: points.length + 1}, (_, i) => ringAt(i));
  const palette = rows.map(row => lighting.map(light => colour(row.thickness_pred, light)));
  return {heightScale, points, rings, palette};
}

function drawTube(project, heightScale, radius, radiusPx, width, height, stride) {
  if (mesh?.heightScale !== heightScale) mesh = prepareMesh(heightScale);
  const {points, rings, palette} = mesh;
  const spans = [];
  const projected = new Map();
  let start = 0;
  while (start < points.length) {
    if (!points[start]) {
      start++;
      continue;
    }
    let end = start + 1;
    while (end < start + stride && points[end]) end++;
    const centreA = project(...rings[start].centre), centreB = project(...rings[end].centre);
    if (!(Math.max(centreA[0], centreB[0]) < -radiusPx || Math.min(centreA[0], centreB[0]) > width + radiusPx ||
        Math.max(centreA[1], centreB[1]) < -radiusPx || Math.min(centreA[1], centreB[1]) > height + radiusPx)) {
      spans.push({start, end, centreA, centreB, depth: centreA[2] + centreB[2]});
    }
    start = end;
  }
  const vertices = index => {
    if (!projected.has(index)) {
      const {centre, offsets} = rings[index];
      projected.set(index, offsets.map(([x, z, y]) =>
        project(centre[0] + x * radius, centre[1] + z * radius, centre[2] + y * radius)));
    }
    return projected.get(index);
  };
  const quad = (a, b, c, d, fill) => {
    context.beginPath();
    context.moveTo(a[0], a[1]);
    context.lineTo(b[0], b[1]);
    context.lineTo(c[0], c[1]);
    context.lineTo(d[0], d[1]);
    context.closePath();
    context.fillStyle = fill;
    context.strokeStyle = fill;
    context.fill();
    context.stroke();
  };
  const inset = (p, o) => [o[0] + .66 * (p[0] - o[0]), o[1] + .66 * (p[1] - o[1])];
  spans.sort((a, b) => a.depth - b.depth);
  context.lineWidth = .65;
  for (const {start, end, centreA, centreB, depth} of spans) {
    const a = vertices(start), b = vertices(end);
    const colours = palette[Math.floor((start + end - 1) / 2)];
    const frontCap = !points[start - 1] && centreA[2] >= centreB[2];
    const rearCap = !points[end] && centreB[2] >= centreA[2];
    for (let i = 0; i < a.length; i++) {
      const j = (i + 1) % a.length;
      const facing = (a[i][2] + a[j][2] + b[i][2] + b[j][2] - 2 * depth) / (4 * radius);
      if (facing > -.02) {
        const upward = (centreA[1] + centreB[1] - (a[i][1] + a[j][1] + b[i][1] + b[j][1]) / 2) / (2 * radiusPx);
        const shade = clamp(Math.round(3 + 2.1 * facing + 2.5 * upward), 0, lighting.length - 1);
        quad(a[i], a[j], b[j], b[i], colours[shade]);
      }
      if (frontCap) quad(a[i], a[j], inset(a[j], centreA), inset(a[i], centreA), colours[4]);
      if (rearCap) quad(b[j], b[i], inset(b[i], centreB), inset(b[j], centreB), colours[4]);
    }
  }
}

function drawScene() {
  frame = 0;
  if (!context) return;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (width < 1 || height < 1) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const bufferWidth = Math.round(width * pixelRatio), bufferHeight = Math.round(height * pixelRatio);
  if (canvas.width !== bufferWidth) canvas.width = bufferWidth;
  if (canvas.height !== bufferHeight) canvas.height = bufferHeight;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!bounds) {
    mapScale.hidden = true;
    context.textAlign = "center";
    context.fillStyle = "#a9bdc0";
    context.font = "14px system-ui";
    context.fillText("Координаты трассы не заданы", width / 2, height / 2);
    return;
  }

  const heightScale = Number(byId("height-scale").value);
  const [east, north, elevation] = route[view.target] ?? bounds.mid;
  const span = Math.max(1, hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1],
    (bounds.max[2] - bounds.min[2]) * heightScale));
  const baseScale = Math.min(width * .84, height * 1.59) / span;
  const scale = baseScale * view.zoom;
  const sy = Math.sin(view.yaw), cy = Math.cos(view.yaw);
  const sp = Math.sin(view.pitch), cp = Math.cos(view.pitch);
  function project(x, z, h) {
    const dx = x - east, dz = z - north, dy = h - elevation * heightScale;
    const forward = dx * sy + dz * cy;
    const depth = forward * cp + dy * sp;
    return [width / 2 + view.offsetX + scale * (dx * cy - dz * sy),
      height * .49 + view.offsetY - scale * (dy * cp - forward * sp), depth];
  }

  const gridStep = niceDistance(Math.min(width, height) * .20 / scale);
  const gridY = (view.target === null ? bounds.min[2] : elevation) * heightScale - 21 / scale;
  const [originX, originY] = project(east, north, gridY);
  function unprojectGrid(x, y) {
    const sideways = (x - originX) / scale, forward = (y - originY) / (scale * sp);
    return [east + sideways * cy + forward * sy, north - sideways * sy + forward * cy];
  }
  drawGrid((x, y) => project(x, y, gridY), unprojectGrid, gridStep, width, height, Math.abs(cy) >= Math.abs(sy));

  screenPoints = route.map(point => point && project(point[0], point[1], point[2] * heightScale));
  const radiusPx = clamp(Math.min(width, height) * .038 * Math.sqrt(view.zoom), 7, 26);
  hitRadius = radiusPx + 8;
  context.lineJoin = "round";
  // At overview scale multiple cells share a pixel. Restore one span per cell as the route fills the view.
  const stride = 2 ** Math.max(0, Math.ceil(Math.log2(3 * rows.length / (baseScale * span * view.zoom))));
  drawTube(project, heightScale, radiusPx / scale, radiusPx, width, height, stride);
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
    if (!point || point[0] < 0 || point[0] > width || point[1] < 0 || point[1] > height) continue;
    const bucket = Math.round(point[0] / 13) + "/" + Math.round(point[1] / 13);
    if (occupied.has(bucket)) continue;
    occupied.add(bucket);
    pointMarker(point, 2.2, "#ffd28c", "#33291e");
  }
  const selected = screenPoints[chosen[activeCase]];
  if (selected) {
    pointMarker(selected, radiusPx + 4, "#f7fff9");
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

let expandedInPage = false;
let switchingFullscreen = false;
function syncFullscreen() {
  const expanded = expandedInPage || document.fullscreenElement === routePanel;
  routePanel.classList.toggle("expanded", expanded);
  document.body.classList.toggle("expanded-route", expanded);
  fullscreenButton.setAttribute("aria-pressed", String(expanded));
  fullscreenButton.title = expanded ? "Свернуть 3D-вид" : "Развернуть 3D-вид";
  redrawScene();
}

fullscreenButton.addEventListener("click", async () => {
  if (switchingFullscreen) return;
  switchingFullscreen = true;
  try {
    if (document.fullscreenElement === routePanel) {
      await document.exitFullscreen();
    } else if (expandedInPage) {
      expandedInPage = false;
    } else {
      try {
        if (routePanel.requestFullscreen) await routePanel.requestFullscreen();
        else expandedInPage = true;
      } catch {
        expandedInPage = true;
      }
    }
  } finally {
    switchingFullscreen = false;
    syncFullscreen();
  }
});
document.addEventListener("fullscreenchange", syncFullscreen);
document.addEventListener("keydown", event => {
  if (event.key !== "Escape" || !expandedInPage) return;
  expandedInPage = false;
  syncFullscreen();
  fullscreenButton.focus();
});

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

function renderScores(warm, closed, open) {
  const charts = byId("score-charts");
  charts.replaceChildren();
  for (const [key, title, direction] of [
    ["pearson_squared", "Pearson²", "выше — лучше"], ["mae_mm", "MAE, мм", "ниже — лучше"]
  ]) {
    const group = document.createElement("div");
    group.className = "score-metric";
    const heading = document.createElement("h4");
    const hint = document.createElement("span");
    hint.textContent = direction;
    heading.append(title + " · ", hint);
    group.append(heading);
    const max = key === "pearson_squared" ? 1 : Math.ceil(Math.max(1, closed?.[key] ?? 0, open?.[key] ?? 0));
    for (const [style, label, metrics] of [
      ["closed", "Закрытая", closed], ["open", "Открытая", open]
    ]) {
      const value = metrics?.[key];
      const row = document.createElement("div");
      row.className = "score-row " + style;
      const tag = document.createElement("span");
      tag.textContent = label;
      const track = document.createElement("div");
      track.className = "score-track";
      track.setAttribute("aria-hidden", "true");
      const bar = document.createElement("i");
      bar.style.width = (Number.isFinite(value) ? 100 * clamp(value / max, 0, 1) : 0) + "%";
      track.append(bar);
      const output = document.createElement("output");
      output.textContent = fmt(value, 3);
      row.append(tag, track, output);
      group.append(row);
    }
    const axis = document.createElement("div");
    axis.className = "score-axis";
    axis.setAttribute("aria-hidden", "true");
    for (const value of [0, max]) {
      const tick = document.createElement("span");
      tick.textContent = fmt(value);
      axis.append(tick);
    }
    group.append(axis);
    charts.append(group);
  }
  setText("closed-note", closed ? "Закрытая — оценка организаторов для файла сдачи." : "Закрытая оценка для этого файла отсутствует.");
  setText("open-note", warm ? "Открытая — поздние ВТД соседних участков." : "Открытая — среднее по кампаниям и магистралям.");
}

function renderScatter(rows, warm) {
  if (scatter.clientWidth < 80) return;
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

function renderQuality() {
  const warm = activeCase === "case1";
  const split = warm ? "late_2023" : "holdout_main";
  const observations = data.observations.filter(row => row.case === activeCase && row.split === split);
  const closed = data.summary.closed_validation?.[activeCase];
  const open = warm
    ? data.metrics.find(row => row.case === activeCase && row.model === "direct_blend" && row.split === split)
    : data.summary.case2_macro_validation.find(row => row.model === "xgboost_knn" && row.seed === data.summary.seed);
  setText("forecast-model", "Итоговая модель · " + forecastModels[activeCase]);
  setText("scatter-meta", fmt(observations.length) + " меток · " +
    (warm ? "поздние ВТД" : "MAIN-2 и MAIN-3 по очереди вне обучения"));
  renderScores(warm, closed, open);
  renderScatter(observations, warm);
}

function selectSegment(index) {
  if (!Number.isFinite(index)) index = chosen[activeCase] ?? 0;
  index = clamp(Math.round(index), 0, rows.length - 1);
  chosen[activeCase] = index;
  const row = rows[index];
  byId("focus-segment").disabled = !route[index];
  if (view.follow && route[index]) view.target = index;
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
  const thicknesses = rows.map(row => row.thickness_pred);
  minThickness = Math.min(...thicknesses);
  maxThickness = Math.max(...thicknesses);
  setText("thickness-min", fmt(minThickness, 2));
  setText("thickness-max", fmt(maxThickness, 2));
  route = data.routes[rows[0].id] ?? [];
  bounds = routeBounds(route);
  mesh = null;
  screenPoints = [];
  priorityRows = rows.filter(row => row.rank !== null).sort((a, b) => a.rank - b.rank);
  const priority = byId("priority");
  priority.replaceChildren(new Option("Выбрать · " + fmt(priorityRows.length) + " сегм.", ""));
  for (const row of priorityRows) priority.add(new Option(
    "№" + row.rank + " · сегм. " + row.segment_id + " · " + mm(row.thickness_pred), row.segment_id));
  setText("ranking-model", rankingModels[next] + " формирует список приоритетов.");
  const date = rows[0].target_date.slice(0, 10).split("-").reverse().join(".");
  setText("route-meta", rows[0].id + " / " + date + " / " + fmt(rows[rows.length - 1].end_m / 1000, 2) + " км");
  setText("route-coverage", bounds && bounds.count < rows.length ? fmt(bounds.count) + " из " + fmt(rows.length) + " сегм. с координатами" : "");
  byId("segment").max = rows.length - 1;
  byId("anchor-key").hidden = !rows.some(row => row.anchor_observed);
  for (const button of caseTabs) {
    const selected = button.dataset.case === next;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  byId("case-panel").setAttribute("aria-labelledby", "tab-" + next);
  Object.assign(view, initialView);
  if (next === "case2") view.yaw -= .26;
  selectSegment(chosen[next] ?? priorityRows[0]?.segment_id ?? 0);
  renderQuality();
}

function nearestSegment(x, y) {
  let best = hitRadius ** 2, nearest = null;
  screenPoints.forEach((point, i) => {
    if (!point) return;
    const distance = (point[0] - x) ** 2 + (point[1] - y) ** 2;
    if (distance < best) {
      nearest = i;
      best = distance;
    }
  });
  return nearest;
}

function canvasPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function zoom(factor, position) {
  const next = clamp(view.zoom * factor, .45, 640);
  if (position && next !== view.zoom) {
    const [x, y] = position;
    const index = frame ? null : nearestSegment(x, y);
    const cx = canvas.clientWidth / 2, cy = canvas.clientHeight * .49;
    const point = index === null ? [cx + view.offsetX, cy + view.offsetY] : screenPoints[index];
    const ratio = next / view.zoom;
    view.offsetX = x - cx + (point[0] - x) * ratio;
    view.offsetY = y - cy + (point[1] - y) * ratio;
    if (index !== null) view.target = index;
    view.follow = false;
  }
  view.zoom = next;
  redrawScene();
}

function focusSegment(index = chosen[activeCase]) {
  if (!route[index]) return;
  const routeLength = rows[rows.length - 1].end_m;
  Object.assign(view, {target: index, offsetX: 0, offsetY: 0, follow: true,
    zoom: Math.max(1, routeLength / 140)});
  selectSegment(index);
}

function stepSegment(delta) {
  selectSegment(chosen[activeCase] + delta);
}

byId("height-scale").addEventListener("change", redrawScene);
byId("focus-segment").addEventListener("click", () => focusSegment());
byId("prev-segment").addEventListener("click", () => stepSegment(-1));
byId("next-segment").addEventListener("click", () => stepSegment(1));
byId("profile-window").addEventListener("change", renderProfile);
byId("segment").addEventListener("change", event => selectSegment(event.target.valueAsNumber));
byId("priority").addEventListener("change", event => {
  if (event.target.value) selectSegment(Number(event.target.value));
});
byId("zoom-in").addEventListener("click", () => zoom(1.23));
byId("zoom-out").addEventListener("click", () => zoom(1 / 1.23));
byId("reset-view").addEventListener("click", () => {
  Object.assign(view, initialView);
  if (activeCase === "case2") view.yaw -= .26;
  redrawScene();
});
canvas.addEventListener("keydown", event => {
  const key = event.key;
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"].includes(key)) return;
  event.preventDefault();
  if (key === "ArrowLeft") view.yaw -= .1;
  if (key === "ArrowRight") view.yaw += .1;
  if (key === "ArrowUp") view.pitch = clamp(view.pitch + .1, minPitch, 1.4);
  if (key === "ArrowDown") view.pitch = clamp(view.pitch - .1, minPitch, 1.4);
  if (key === "+" || key === "=") zoom(1.23);
  if (key === "-") zoom(1 / 1.23);
  if (key === "Home") byId("reset-view").click();
  redrawScene();
});
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 200 : 1);
  zoom(Math.exp(clamp(-delta * .0025, -.5, .5)), canvasPosition(event));
}, {passive: false});
canvas.addEventListener("dblclick", event => {
  const index = nearestSegment(...canvasPosition(event));
  if (index !== null) focusSegment(index);
});
const pointers = new Map();
let moved = false;
canvas.addEventListener("pointerdown", event => {
  if (event.button !== 0 || pointers.size >= 2) return;
  moved = pointers.size > 0;
  pointers.set(event.pointerId, canvasPosition(event));
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  const before = pointers.get(event.pointerId);
  if (!before) return;
  const current = canvasPosition(event);
  const dx = current[0] - before[0], dy = current[1] - before[1];
  if (!moved && hypot(dx, dy) < 4) return;
  moved = true;
  pointers.set(event.pointerId, current);
  if (pointers.size === 2) {
    const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)[1];
    const previousDistance = hypot(before[0] - other[0], before[1] - other[1]);
    if (previousDistance < 1) return;
    const distance = hypot(current[0] - other[0], current[1] - other[1]);
    zoom(distance / previousDistance, [(before[0] + other[0]) / 2, (before[1] + other[1]) / 2]);
    view.offsetX += dx / 2;
    view.offsetY += dy / 2;
  } else if (event.shiftKey) {
    view.offsetX += dx;
    view.offsetY += dy;
  } else {
    view.yaw += dx * .009;
    view.pitch = clamp(view.pitch + dy * .009, minPitch, 1.4);
  }
  redrawScene();
});
canvas.addEventListener("pointerup", event => {
  if (!pointers.delete(event.pointerId) || moved) return;
  const nearest = nearestSegment(...canvasPosition(event));
  if (nearest !== null) selectSegment(nearest);
});
canvas.addEventListener("lostpointercapture", event => pointers.delete(event.pointerId));
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
for (const [index, button] of caseTabs.entries()) {
  button.addEventListener("click", () => switchCase(button.dataset.case));
  button.addEventListener("keydown", event => {
    const target = {ArrowLeft: (index - 1 + caseTabs.length) % caseTabs.length,
      ArrowRight: (index + 1) % caseTabs.length, Home: 0, End: caseTabs.length - 1}[event.key];
    if (target === undefined) return;
    event.preventDefault();
    caseTabs[target].focus();
    switchCase(caseTabs[target].dataset.case);
  });
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
