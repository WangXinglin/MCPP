const state = {
  manifest: null,
  axes: null,
  queries: [],
  selectedQueryIndex: 0,
  matrix: null,
  budgetIndex: 0,
  deadlineIndex: 0,
  filter: "",
};

const els = {
  dataBadge: document.querySelector("#dataBadge"),
  querySearch: document.querySelector("#querySearch"),
  queryCount: document.querySelector("#queryCount"),
  queryList: document.querySelector("#queryList"),
  selectedTitle: document.querySelector("#selectedTitle"),
  selectedId: document.querySelector("#selectedId"),
  budgetSlider: document.querySelector("#budgetSlider"),
  deadlineSlider: document.querySelector("#deadlineSlider"),
  budgetValue: document.querySelector("#budgetValue"),
  deadlineValue: document.querySelector("#deadlineValue"),
  budgetMin: document.querySelector("#budgetMin"),
  budgetMid: document.querySelector("#budgetMid"),
  budgetMax: document.querySelector("#budgetMax"),
  deadlineMin: document.querySelector("#deadlineMin"),
  deadlineMid: document.querySelector("#deadlineMid"),
  deadlineMax: document.querySelector("#deadlineMax"),
  successRate: document.querySelector("#successRate"),
};

function formatBudget(value) {
  return value < 1 ? `$${value.toFixed(2)}` : `$${value.toFixed(value >= 10 ? 1 : 2)}`;
}

function formatDeadline(seconds) {
  if (seconds < 90) return `${Math.round(seconds)} sec`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  const hours = seconds / 3600;
  return `${hours.toFixed(hours >= 2 ? 1 : 2)} hr`;
}

function formatRate(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${Math.round(value * 100)}%`;
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load ${path}`);
  return response.json();
}

async function loadMatrix(query) {
  state.matrix = await loadJson(`./data/${query.matrix_path}`);
  renderAll();
}

function filteredQueries() {
  const term = state.filter.trim().toLowerCase();
  if (!term) return state.queries;
  return state.queries.filter((query) => {
    return `${query.query_id} ${query.problem_id ?? ""} ${query.title} ${query.source ?? ""}`.toLowerCase().includes(term);
  });
}

function renderQueryList() {
  const visible = filteredQueries();
  els.queryList.innerHTML = "";
  els.queryCount.textContent = `Showing ${visible.length} of ${state.queries.length} queries`;

  visible.forEach((query) => {
    const originalIndex = state.queries.findIndex((item) => item.query_id === query.query_id);
    const button = document.createElement("button");
    button.className = `query-item ${originalIndex === state.selectedQueryIndex ? "active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <span>${query.query_id}</span>
      <strong>${query.title}</strong>
    `;
    button.addEventListener("click", async () => {
      state.selectedQueryIndex = originalIndex;
      state.matrix = null;
      renderQueryList();
      renderLoadingQuery(query);
      await loadMatrix(query);
    });
    els.queryList.appendChild(button);
  });
}

function renderAxes() {
  const budgets = state.axes.budgets;
  const deadlines = state.axes.deadlines;
  els.budgetSlider.max = String(budgets.length - 1);
  els.deadlineSlider.max = String(deadlines.length - 1);
  els.budgetSlider.value = String(state.budgetIndex);
  els.deadlineSlider.value = String(state.deadlineIndex);

  els.budgetMin.textContent = formatBudget(budgets[0]);
  els.budgetMid.textContent = formatBudget(budgets[Math.floor((budgets.length - 1) / 2)]);
  els.budgetMax.textContent = formatBudget(budgets[budgets.length - 1]);

  els.deadlineMin.textContent = formatDeadline(deadlines[0]);
  els.deadlineMid.textContent = formatDeadline(deadlines[Math.floor((deadlines.length - 1) / 2)]);
  els.deadlineMax.textContent = formatDeadline(deadlines[deadlines.length - 1]);
}

function renderLoadingQuery(query) {
  els.selectedTitle.textContent = query.title;
  els.selectedId.textContent = query.query_id;
  els.successRate.textContent = "--";
}

function selectedCell() {
  if (!state.matrix) return { rate: null, trials: 0, status: "loading" };
  const rate = getCellRate(state.budgetIndex, state.deadlineIndex);
  const status = getCellStatus(state.budgetIndex, state.deadlineIndex);
  return {
    rate,
    trials: state.matrix.n_trials[state.budgetIndex]?.[state.deadlineIndex] ?? 0,
    status,
  };
}

function getCellRate(budgetIndex, deadlineIndex) {
  if (state.matrix?.success_pct) {
    const value = state.matrix.success_pct[budgetIndex]?.[deadlineIndex];
    return value === null || value === undefined ? null : value / 100;
  }
  const value = state.matrix?.success_rate?.[budgetIndex]?.[deadlineIndex];
  return value === undefined ? null : value;
}

function getCellStatus(budgetIndex, deadlineIndex) {
  const status = state.matrix?.status;
  if (!status) return "missing";
  if (Array.isArray(status[budgetIndex])) {
    return status[budgetIndex]?.[deadlineIndex] ?? "missing";
  }
  const missingKey = `${budgetIndex},${deadlineIndex}`;
  return status.missing?.includes(missingKey) ? "missing" : "complete";
}

function renderResult() {
  const query = state.queries[state.selectedQueryIndex];
  const budget = state.axes.budgets[state.budgetIndex];
  const deadline = state.axes.deadlines[state.deadlineIndex];
  const cell = selectedCell();

  els.selectedTitle.textContent = state.matrix?.title ?? query.title;
  els.selectedId.textContent = query.query_id;
  els.budgetValue.textContent = formatBudget(budget);
  els.deadlineValue.textContent = formatDeadline(deadline);
  els.successRate.textContent = formatRate(cell.rate);
  els.successRate.classList.toggle("missing", cell.status !== "complete");
}

function renderAll() {
  renderAxes();
  renderQueryList();
  renderResult();
}

async function init() {
  try {
    const [manifest, axes, queries] = await Promise.all([
      loadJson("./data/manifest.json"),
      loadJson("./data/axes.json"),
      loadJson("./data/queries.json"),
    ]);
    state.manifest = manifest;
    state.axes = axes;
    state.queries = queries;
    els.dataBadge.textContent = `${manifest.query_count} queries`;
    renderAxes();
    renderQueryList();
    renderLoadingQuery(queries[0]);
    await loadMatrix(queries[0]);
  } catch (error) {
    console.error(error);
    els.dataBadge.textContent = "Data load failed";
    els.selectedTitle.textContent = "Could not load demo data";
  }
}

els.querySearch.addEventListener("input", (event) => {
  state.filter = event.target.value;
  renderQueryList();
});

els.budgetSlider.addEventListener("input", (event) => {
  state.budgetIndex = Number(event.target.value);
  renderResult();
});

els.deadlineSlider.addEventListener("input", (event) => {
  state.deadlineIndex = Number(event.target.value);
  renderResult();
});

init();
