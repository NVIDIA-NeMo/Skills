/**
 * AceProof TTS Monitor - Enhanced Frontend
 * Features: KaTeX math rendering, tabs, modals, filtering, sorting
 */

// ============================================
// API & Utilities
// ============================================

const api = async (path) => {
  const res = await fetch(path);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
};

const formatScore = (score) => {
  if (score === null || score === undefined) return "--";
  return score.toFixed(3);
};

const formatDuration = (seconds) => {
  if (!seconds) return "--";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

const escapeHtml = (text) => {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
};

const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error("Copy failed:", err);
    return false;
  }
};

// ============================================
// Math Rendering with KaTeX
// ============================================

const renderMath = (element) => {
  if (typeof renderMathInElement === "function") {
    try {
      renderMathInElement(element, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        trust: true,
        strict: false,
      });
    } catch (err) {
      console.warn("KaTeX render warning:", err);
    }
  }
};

const setMathContent = (element, text) => {
  element.textContent = text || "";
  renderMath(element);
};

// ============================================
// State Management
// ============================================

const state = {
  problems: [],
  filteredProblems: [],
  selected: null,
  summary: null,
  stageCache: {},
  activeStageButton: null,
  currentTab: "proofs",
  currentFilter: "all",
  currentSort: "score-desc",
  currentProofData: null,
};

// ============================================
// DOM Elements
// ============================================

const $ = (id) => document.getElementById(id);

const elements = {
  // Header
  runDir: $("runDir"),
  statProblems: $("statProblems"),
  statProofs: $("statProofs"),
  statRounds: $("statRounds"),
  refreshBtn: $("refreshBtn"),
  themeToggle: $("themeToggle"),

  // Error
  errorBanner: $("errorBanner"),

  // Progress
  progressBar: $("progressBar"),
  progressText: $("progressText"),
  progressPercent: $("progressPercent"),
  timeline: $("timeline"),

  // Problems
  problemList: $("problemList"),
  problemCount: $("problemCount"),
  searchInput: $("searchInput"),

  // Detail
  detailView: $("detailView"),
  detailEmpty: $("detailEmpty"),
  detailTitle: $("detailTitle"),
  detailProofCount: $("detailProofCount"),
  detailBestScore: $("detailBestScore"),
  breadcrumbProblem: $("breadcrumbProblem"),

  // Problem content
  problemText: $("problemText"),
  copyProblemBtn: $("copyProblemBtn"),

  // Trend
  trendChart: $("trendChart"),

  // Distribution
  scoreDistChart: $("scoreDistChart"),
  scoreDistTotal: $("scoreDistTotal"),
  scoreDistPerfect: $("scoreDistPerfect"),

  // Proofs
  proofList: $("proofList"),
  proofListCount: $("proofListCount"),
  proofSortSelect: $("proofSortSelect"),

  // Stages
  stageExplorer: $("stageExplorer"),
  stageDetail: $("stageDetail"),

  // Tabs
  tabProofs: $("tabProofs"),
  tabStages: $("tabStages"),

  // Modal
  proofModal: $("proofModal"),
  modalTitle: $("modalTitle"),
  modalProofContent: $("modalProofContent"),
  modalMeta: $("modalMeta"),
  copyProofBtn: $("copyProofBtn"),
  closeModalBtn: $("closeModalBtn"),
};

// ============================================
// Error Handling
// ============================================

const showError = (message) => {
  if (!message) return;
  elements.errorBanner.textContent = message;
  elements.errorBanner.classList.remove("hidden");
  setTimeout(() => clearError(), 10000);
};

const clearError = () => {
  elements.errorBanner.textContent = "";
  elements.errorBanner.classList.add("hidden");
};

// ============================================
// Theme Toggle
// ============================================

const initTheme = () => {
  const saved = localStorage.getItem("theme");
  if (saved) {
    document.documentElement.setAttribute("data-theme", saved);
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
  }
};

const toggleTheme = () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
};

// ============================================
// Timeline Rendering
// ============================================

const renderTimeline = (stages) => {
  elements.timeline.innerHTML = "";
  let done = 0;
  let total = 0;

  const template = document.getElementById("timelineItemTemplate");

  stages.forEach((round) => {
    const stageNames = Object.keys(round).filter((k) => k !== "round_idx");
    stageNames.forEach((stage) => {
      total += 1;
      const status = round[stage];
      const isDone = status.done_exists;
      const isRunning = status.output_exists && !isDone;

      if (isDone) done += 1;

      const node = template.content.cloneNode(true);
      const item = node.querySelector(".timeline-item");
      const title = node.querySelector(".timeline-title");
      const statusEl = node.querySelector(".timeline-status");
      const duration = node.querySelector(".timeline-duration");

      item.classList.add(isDone ? "done" : isRunning ? "running" : "pending");
      title.textContent = `R${round.round_idx} · ${stage}`;

      const statusText = isDone ? "Done" : isRunning ? "Running" : "Pending";
      statusEl.textContent = statusText;
      statusEl.className = `timeline-status ${isDone ? "done" : isRunning ? "running" : "pending"}`;

      duration.textContent = formatDuration(status.duration_sec);

      elements.timeline.appendChild(node);
    });
  });

  const progress = total === 0 ? 0 : (done / total) * 100;
  elements.progressBar.style.width = `${progress}%`;
  elements.progressText.textContent = `${done}/${total} stages`;
  elements.progressPercent.textContent = `${Math.round(progress)}%`;
};

// ============================================
// Problems List
// ============================================

const applyFiltersAndSort = () => {
  let filtered = [...state.problems];

  // Apply filter
  switch (state.currentFilter) {
    case "high":
      filtered = filtered.filter((p) => p.best_meanscore >= 0.8);
      break;
    case "low":
      filtered = filtered.filter((p) => p.best_meanscore < 0.5);
      break;
    case "recent":
      const maxRound = Math.max(...filtered.map((p) => p.latest_round));
      filtered = filtered.filter((p) => p.latest_round === maxRound);
      break;
  }

  // Apply search
  const query = elements.searchInput.value.trim().toLowerCase();
  if (query) {
    filtered = filtered.filter((p) =>
      `${p.source_name}/${p.problem_idx}`.toLowerCase().includes(query)
    );
  }

  state.filteredProblems = filtered;
  renderProblems(filtered);
};

const renderProblems = (list) => {
  const template = document.getElementById("problemTemplate");
  elements.problemList.innerHTML = "";
  elements.problemCount.textContent = list.length;

  list.forEach((problem) => {
    const node = template.content.cloneNode(true);
    const item = node.querySelector(".problem-item");

    item.dataset.source = problem.source_name;
    item.dataset.problem = problem.problem_idx;

    // Highlight active
    if (
      state.selected &&
      state.selected.source_name === problem.source_name &&
      state.selected.problem_idx === problem.problem_idx
    ) {
      item.classList.add("active");
    }

    node.querySelector(".problem-id").textContent =
      `${problem.source_name}/${problem.problem_idx}`;
    node.querySelector(".problem-meta").textContent =
      `${problem.num_proofs} proofs · R${problem.latest_round}`;
    node.querySelector(".score-value").textContent =
      formatScore(problem.best_meanscore);
    node.querySelector(".round-badge").textContent = `Round ${problem.latest_round}`;

    elements.problemList.appendChild(node);
  });
};

// ============================================
// Trend Chart
// ============================================

const renderTrend = (trend) => {
  elements.trendChart.innerHTML = "";

  if (!trend || trend.length === 0) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "150");
    text.setAttribute("y", "50");
    text.setAttribute("fill", "var(--text-tertiary)");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "12");
    text.textContent = "No trend data";
    elements.trendChart.appendChild(text);
    return;
  }

  const width = 300;
  const height = 140;
  const padding = { top: 15, right: 15, bottom: 25, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const best = trend.map((d) => d.best_meanscore);
  const avg = trend.map((d) => d.avg_meanscore);
  const allScores = [...best, ...avg];
  const maxY = Math.max(...allScores, 0.1);
  const minY = Math.min(...allScores, 0);

  const scaleX = (i) =>
    padding.left + (i / Math.max(1, trend.length - 1)) * chartWidth;
  const scaleY = (v) =>
    padding.top + (1 - (v - minY) / (maxY - minY || 1)) * chartHeight;

  // Grid lines
  const gridGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (i / 4) * chartHeight;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--border-subtle)");
    line.setAttribute("stroke-width", "1");
    gridGroup.appendChild(line);

    // Y-axis labels
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const value = maxY - (i / 4) * (maxY - minY);
    label.setAttribute("x", padding.left - 5);
    label.setAttribute("y", y + 3);
    label.setAttribute("fill", "var(--text-tertiary)");
    label.setAttribute("font-size", "9");
    label.setAttribute("text-anchor", "end");
    label.textContent = value.toFixed(2);
    gridGroup.appendChild(label);
  }
  elements.trendChart.appendChild(gridGroup);

  // X-axis labels (rounds)
  trend.forEach((d, i) => {
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", scaleX(i));
    label.setAttribute("y", height - 5);
    label.setAttribute("fill", "var(--text-tertiary)");
    label.setAttribute("font-size", "9");
    label.setAttribute("text-anchor", "middle");
    label.textContent = `R${d.round_idx}`;
    elements.trendChart.appendChild(label);
  });

  // Draw lines
  const drawLine = (values, color, dashed = false) => {
    if (values.length < 2) {
      // Draw points for single values
      values.forEach((v, i) => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", scaleX(i));
        circle.setAttribute("cy", scaleY(v));
        circle.setAttribute("r", "4");
        circle.setAttribute("fill", color);
        elements.trendChart.appendChild(circle);
      });
      return;
    }

    const points = values.map((v, i) => `${scaleX(i)},${scaleY(v)}`).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", points);
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", color);
    poly.setAttribute("stroke-width", "2");
    if (dashed) poly.setAttribute("stroke-dasharray", "4,2");
    elements.trendChart.appendChild(poly);

    // Add dots
    values.forEach((v, i) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", scaleX(i));
      circle.setAttribute("cy", scaleY(v));
      circle.setAttribute("r", "3");
      circle.setAttribute("fill", color);
      elements.trendChart.appendChild(circle);
    });
  };

  drawLine(avg, "var(--chart-avg)", true);
  drawLine(best, "var(--chart-best)");
};

// ============================================
// Score Distribution Chart
// ============================================

const buildDistributionFromProofs = (proofs, binCount = 10) => {
  if (!proofs || proofs.length === 0) {
    return { bins: [], total: 0, perfect: 0 };
  }
  const counts = Array(binCount).fill(0);
  let total = 0;
  let perfect = 0;
  proofs.forEach((proof) => {
    let score = proof?.meanscore;
    if (typeof score !== "number") return;
    score = Math.max(0, Math.min(1, score));
    if (score >= 1) perfect += 1;
    total += 1;
    const idx = Math.min(Math.floor(score * binCount), binCount - 1);
    counts[idx] += 1;
  });

  const bins = counts.map((count, idx) => {
    const min = idx / binCount;
    const max = (idx + 1) / binCount;
    return {
      min,
      max,
      label: `${min.toFixed(1)}-${max.toFixed(1)}`,
      count,
    };
  });

  return { bins, total, perfect };
};

const renderScoreDistribution = (dist) => {
  const chart = elements.scoreDistChart;
  if (!chart) return;
  chart.innerHTML = "";

  if (!dist || !dist.bins || dist.bins.length === 0) {
    if (elements.scoreDistTotal) elements.scoreDistTotal.textContent = "-- proofs";
    if (elements.scoreDistPerfect) elements.scoreDistPerfect.textContent = "-- score=1";
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "130");
    text.setAttribute("y", "80");
    text.setAttribute("fill", "var(--text-tertiary)");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "12");
    text.textContent = "No distribution data";
    chart.appendChild(text);
    return;
  }

  if (elements.scoreDistTotal) {
    elements.scoreDistTotal.textContent = `${dist.total} proofs`;
  }
  if (elements.scoreDistPerfect) {
    elements.scoreDistPerfect.textContent = `${dist.perfect} score=1`;
  }

  const width = 280;
  const height = 140;
  const padding = { top: 10, right: 10, bottom: 22, left: 32 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const maxCount = Math.max(...dist.bins.map((b) => b.count), 1);
  const barWidth = chartWidth / dist.bins.length;

  for (let i = 0; i <= 3; i++) {
    const y = padding.top + (i / 3) * chartHeight;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--border-subtle)");
    line.setAttribute("stroke-width", "1");
    chart.appendChild(line);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const value = Math.round(maxCount - (i / 3) * maxCount);
    label.setAttribute("x", padding.left - 6);
    label.setAttribute("y", y + 3);
    label.setAttribute("fill", "var(--text-tertiary)");
    label.setAttribute("font-size", "9");
    label.setAttribute("text-anchor", "end");
    label.textContent = value;
    chart.appendChild(label);
  }

  dist.bins.forEach((bin, idx) => {
    const barHeight = (bin.count / maxCount) * chartHeight;
    const x = padding.left + idx * barWidth + barWidth * 0.15;
    const y = padding.top + chartHeight - barHeight;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", barWidth * 0.7);
    rect.setAttribute("height", barHeight);
    rect.setAttribute("rx", "2");
    rect.setAttribute("class", "distribution-bar");
    chart.appendChild(rect);
  });

  const labelStart = document.createElementNS("http://www.w3.org/2000/svg", "text");
  labelStart.setAttribute("x", padding.left);
  labelStart.setAttribute("y", height - 8);
  labelStart.setAttribute("fill", "var(--text-tertiary)");
  labelStart.setAttribute("font-size", "9");
  labelStart.setAttribute("text-anchor", "start");
  labelStart.textContent = "0.0";
  chart.appendChild(labelStart);

  const labelMid = document.createElementNS("http://www.w3.org/2000/svg", "text");
  labelMid.setAttribute("x", padding.left + chartWidth / 2);
  labelMid.setAttribute("y", height - 8);
  labelMid.setAttribute("fill", "var(--text-tertiary)");
  labelMid.setAttribute("font-size", "9");
  labelMid.setAttribute("text-anchor", "middle");
  labelMid.textContent = "0.5";
  chart.appendChild(labelMid);

  const labelEnd = document.createElementNS("http://www.w3.org/2000/svg", "text");
  labelEnd.setAttribute("x", width - padding.right);
  labelEnd.setAttribute("y", height - 8);
  labelEnd.setAttribute("fill", "var(--text-tertiary)");
  labelEnd.setAttribute("font-size", "9");
  labelEnd.setAttribute("text-anchor", "end");
  labelEnd.textContent = "1.0";
  chart.appendChild(labelEnd);
};

// ============================================
// Tabs
// ============================================

const switchTab = (tabName) => {
  state.currentTab = tabName;

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });

  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.toggle("active", content.id === `tab${capitalize(tabName)}`);
  });
};

const capitalize = (str) => str.charAt(0).toUpperCase() + str.slice(1);

// ============================================
// Proof Modal
// ============================================

const openProofModal = (proof) => {
  state.currentProofData = proof;
  elements.modalTitle.textContent = `Proof ${proof.proof_id ?? "?"} · Round ${proof.round_idx}`;
  setMathContent(elements.modalProofContent, proof.proof || proof.proof_preview || "No content");
  elements.modalMeta.textContent =
    `Score: ${formatScore(proof.meanscore)} | Self-eval: ${formatScore(proof.self_eval_score)} | Hash: ${proof.proof_hash?.slice(0, 12)}...`;
  elements.proofModal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
};

const closeProofModal = () => {
  elements.proofModal.classList.add("hidden");
  document.body.style.overflow = "";
  state.currentProofData = null;
};

// ============================================
// Proofs Rendering
// ============================================

const sortProofs = (proofs) => {
  const sorted = [...proofs];
  switch (state.currentSort) {
    case "score-desc":
      return sorted.sort((a, b) => b.meanscore - a.meanscore);
    case "score-asc":
      return sorted.sort((a, b) => a.meanscore - b.meanscore);
    case "round-desc":
      return sorted.sort((a, b) => b.round_idx - a.round_idx);
    case "round-asc":
      return sorted.sort((a, b) => a.round_idx - b.round_idx);
    default:
      return sorted;
  }
};

const renderProofs = (proofs) => {
  elements.proofList.innerHTML = "";
  elements.proofListCount.textContent = `${proofs.length} proofs`;

  const template = document.getElementById("proofTemplate");
  const sortedProofs = sortProofs(proofs);

  sortedProofs.forEach((proof) => {
    const node = template.content.cloneNode(true);

    node.querySelector(".proof-title").textContent =
      `Proof ${proof.proof_id ?? "?"} · Round ${proof.round_idx}`;
    node.querySelector(".proof-meta").textContent =
      proof.dep_proof_ids?.length
        ? `Dependencies: ${proof.dep_proof_ids.join(", ")}`
        : "No dependencies";

    node.querySelector(".score-value").textContent = formatScore(proof.meanscore);
    node.querySelector(".self-eval-value").textContent = formatScore(proof.self_eval_score);

    const previewEl = node.querySelector(".proof-preview");
    setMathContent(previewEl, proof.proof_preview || "");

    node.querySelector(".proof-hash").textContent =
      proof.proof_hash ? `#${proof.proof_hash.slice(0, 8)}` : "";

    // Expand button
    const expandBtn = node.querySelector(".expand-btn");
    expandBtn.addEventListener("click", () => openProofModal(proof));

    // Verification button
    const verifyBtn = node.querySelector(".verify-btn");
    const verifications = node.querySelector(".verifications");
    const verList = node.querySelector(".verification-list");
    const verSummary = node.querySelector(".verification-summary");

    verifyBtn.addEventListener("click", async () => {
      if (!verifications.classList.contains("hidden")) {
        verifications.classList.add("hidden");
        verifyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
            <polyline points="22,4 12,14.01 9,11.01"/>
          </svg>
          Load Verifications
        `;
        return;
      }

      verifyBtn.innerHTML = `<span>Loading...</span>`;
      verifyBtn.disabled = true;

      try {
        const resp = await api(
          `/api/verifications?round_idx=${proof.round_idx}&problem_idx=${state.selected.problem_idx}&proof_hash=${proof.proof_hash}&offset=0&limit=64`
        );

        verList.innerHTML = "";
        resp.items.forEach((item, idx) => {
          const div = document.createElement("div");
          div.className = "verification-item";

          const scoreColor =
            item.verification_score >= 0.8
              ? "var(--accent-success)"
              : item.verification_score >= 0.5
                ? "var(--accent-warning)"
                : "var(--accent-danger)";

          div.innerHTML = `
            <strong>
              <span style="color: ${scoreColor}">#${idx + 1}</span>
              · Score: ${item.verification_score}
            </strong>
            <div class="rating-text math-content"></div>
          `;

          const ratingEl = div.querySelector(".rating-text");
          setMathContent(ratingEl, item.rating_text || "No rating text");

          verList.appendChild(div);
        });

        verSummary.textContent = `Avg: ${formatScore(resp.avg_score)} · ${resp.total} ratings`;
        verifications.classList.remove("hidden");

        verifyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6,9 12,15 18,9"/>
          </svg>
          Hide Verifications
        `;
      } catch (err) {
        showError(`Verification load failed: ${err.message}`);
        verifyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
            <polyline points="22,4 12,14.01 9,11.01"/>
          </svg>
          Load Verifications
        `;
      } finally {
        verifyBtn.disabled = false;
      }
    });

    elements.proofList.appendChild(node);
  });
};

// ============================================
// Stage Explorer
// ============================================

const renderStageExplorer = (problem) => {
  elements.stageExplorer.innerHTML = "";
  elements.stageDetail.innerHTML = "";
  state.stageCache = {};

  if (!state.summary || !state.summary.stages) {
    elements.stageExplorer.innerHTML =
      '<div class="muted">No stage data available.</div>';
    return;
  }

  state.summary.stages.forEach((round) => {
    const roundIdx = round.round_idx;
    const stageNames = Object.keys(round).filter((k) => k !== "round_idx");

    stageNames.forEach((stage) => {
      const btn = document.createElement("button");
      btn.className = "stage-chip";
      btn.dataset.round = roundIdx;
      btn.dataset.stage = stage;
      btn.textContent = `R${roundIdx} ${stage}`;
      btn.addEventListener("click", () => loadStage(problem, roundIdx, stage, btn));
      elements.stageExplorer.appendChild(btn);
    });
  });
};

const loadStage = async (problem, roundIdx, stage, button) => {
  const key = `${roundIdx}:${stage}:${problem.source_name}:${problem.problem_idx}`;
  const limit = 25;

  if (!state.stageCache[key]) {
    state.stageCache[key] = { offset: 0, items: [], total: 0, resp: null };
  }
  const cache = state.stageCache[key];

  try {
    clearError();
    const resp = await api(
      `/api/stage?round_idx=${roundIdx}&stage=${stage}&problem_idx=${problem.problem_idx}` +
        `&source_name=${problem.source_name}&offset=${cache.offset}&limit=${limit}&processed=true`
    );

    cache.total = resp.total;
    cache.offset = resp.offset + resp.items.length;
    cache.items = cache.items.concat(resp.items);
    cache.resp = resp;

    if (button) {
      state.activeStageButton = button;
      document.querySelectorAll(".stage-chip").forEach((chip) => {
        chip.classList.toggle("active", chip === button);
      });
    }

    renderStageDetail(resp, cache);
  } catch (err) {
    showError(`Stage load failed: ${err.message}`);
  }
};

const stringifyJson = (obj) => {
  if (!obj) return "(missing)";
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
};

const formatCharCount = (count) => {
  if (!count) return "0 chars";
  if (count < 1000) return `${count} chars`;
  if (count < 1000000) return `${(count / 1000).toFixed(1)}K chars`;
  return `${(count / 1000000).toFixed(2)}M chars`;
};

const renderModelInput = (inputData) => {
  if (!inputData || !inputData.model_input) {
    return '<div class="section-empty">No model input data</div>';
  }

  const modelInput = inputData.model_input;

  if (modelInput.type === "messages") {
    const messages = modelInput.content || [];
    let html = '<div class="message-list">';
    messages.forEach((msg, idx) => {
      const roleClass = msg.role || "unknown";
      html += `
        <div class="message-item" data-msg-idx="${idx}">
          <span class="message-role ${roleClass}">${escapeHtml(msg.role)}</span>
          <div class="message-text math-content">${escapeHtml(msg.content || "")}</div>
        </div>
      `;
    });
    html += "</div>";
    return html;
  } else if (modelInput.type === "prompt") {
    return `<div class="prompt-text math-content">${escapeHtml(modelInput.content || "")}</div>`;
  }

  return '<div class="section-empty">Unknown input format</div>';
};

const renderModelOutput = (outputData, container) => {
  if (!outputData || !outputData.model_output) {
    container.innerHTML = '<div class="section-empty">No model output (pending)</div>';
    return;
  }

  const modelOutput = outputData.model_output;
  const thinkingSection = container.querySelector(".thinking-section");
  const responseSection = container.querySelector(".response-section");
  const responseContent = container.querySelector(".response-content");

  if (modelOutput.has_thinking && modelOutput.thinking) {
    // Show thinking section
    thinkingSection.style.display = "block";
    const thinkingStats = thinkingSection.querySelector(".thinking-stats");
    const thinkingContent = thinkingSection.querySelector(".thinking-content");
    const thinkingContentWrapper = thinkingSection.querySelector(".thinking-content-wrapper");
    const bottomCollapseBtn = thinkingSection.querySelector(".thinking-collapse-btn");

    thinkingStats.textContent = `${modelOutput.thinking_words.toLocaleString()} words`;
    thinkingContent.textContent = modelOutput.thinking;
    renderMath(thinkingContent);

    // Set up toggle for top button
    const toggle = thinkingSection.querySelector(".thinking-toggle");
    const toggleCollapse = () => {
      thinkingSection.classList.toggle("collapsed");
      // Scroll to top of section when collapsing from bottom
      if (thinkingSection.classList.contains("collapsed")) {
        thinkingSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    };

    toggle.addEventListener("click", toggleCollapse);

    // Set up bottom collapse button
    if (bottomCollapseBtn) {
      bottomCollapseBtn.addEventListener("click", toggleCollapse);
    }
  } else {
    // Hide or show notice
    thinkingSection.innerHTML = `
      <div class="no-thinking-notice">No reasoning/thinking block in output</div>
    `;
    thinkingSection.style.display = "block";
  }

  // Render response
  if (modelOutput.response) {
    responseContent.textContent = modelOutput.response;
    renderMath(responseContent);
  } else {
    responseContent.innerHTML = '<span class="muted">(empty response)</span>';
  }
};

const renderStageItemCard = (item, resp) => {
  const template = document.getElementById("stageItemTemplate");
  const node = template.content.cloneNode(true);
  const card = node.querySelector(".stage-item-card");

  // Header
  node.querySelector(".key-label").textContent = `${item.key_field}=`;
  node.querySelector(".key-value").textContent = item.key;

  // Badges
  const inputBadge = node.querySelector(".input-badge");
  const outputBadge = node.querySelector(".output-badge");

  if (!item.has_input) {
    inputBadge.classList.add("missing");
    inputBadge.textContent = "No Input";
  }
  if (!item.has_output) {
    outputBadge.classList.add("missing");
    outputBadge.textContent = "Pending";
    card.classList.add("pending");
  }

  // Metadata
  const metaContainer = node.querySelector(".stage-item-meta");
  const displayFields = item.input?.display || item.output?.display || {};
  let metaHtml = "";

  const priorityFields = ["problem_idx", "proof_id", "round_idx", "meanscore", "verification_score"];
  priorityFields.forEach((field) => {
    if (displayFields[field] !== undefined && displayFields[field] !== null) {
      let value = displayFields[field];
      if (typeof value === "number" && !Number.isInteger(value)) {
        value = value.toFixed(3);
      }
      metaHtml += `<span class="meta-tag"><span class="meta-label">${field}:</span> <span class="meta-value">${escapeHtml(String(value))}</span></span>`;
    }
  });

  metaContainer.innerHTML = metaHtml || '<span class="muted">No metadata</span>';

  // Input/Output Sections
  const inputSection = node.querySelector(".input-section");
  const inputContent = node.querySelector(".input-content");
  const charCount = node.querySelector(".char-count");
  const outputSection = node.querySelector(".output-section");
  const outputFieldName = node.querySelector(".output-field-name");
  const toggleButtons = node.querySelectorAll(".toggle-section-btn");

  const setSectionState = (section, button, collapsed) => {
    if (!section || !button) return;
    section.classList.toggle("collapsed", collapsed);
    button.textContent = collapsed ? "Expand" : "Collapse";
  };

  if (item.input && item.input.model_input) {
    inputContent.innerHTML = renderModelInput(item.input);
    charCount.textContent = formatCharCount(item.input.model_input.char_count);

    // Render math in input after DOM is attached
    setTimeout(() => {
      inputContent.querySelectorAll(".math-content").forEach(renderMath);
    }, 0);
  } else {
    inputContent.innerHTML = '<div class="section-empty">No input data available</div>';
    charCount.textContent = "";
  }

  // Toggle buttons
  const inputToggleBtn = node.querySelector('.toggle-section-btn[data-target="input"]');
  const outputToggleBtn = node.querySelector('.toggle-section-btn[data-target="output"]');

  const inputCharCount = item.input?.model_input?.char_count || 0;
  const outputCharCount = item.output?.model_output?.raw?.length || 0;
  setSectionState(inputSection, inputToggleBtn, inputCharCount > 2000);
  setSectionState(outputSection, outputToggleBtn, outputCharCount > 3000);

  if (inputToggleBtn) {
    inputToggleBtn.addEventListener("click", () => {
      const collapsed = !inputSection.classList.contains("collapsed");
      setSectionState(inputSection, inputToggleBtn, collapsed);
    });
  }

  if (outputToggleBtn) {
    outputToggleBtn.addEventListener("click", () => {
      const collapsed = !outputSection.classList.contains("collapsed");
      setSectionState(outputSection, outputToggleBtn, collapsed);
    });
  }

  // Copy input button
  const copyInputBtn = node.querySelector(".copy-input-btn");
  copyInputBtn.addEventListener("click", async () => {
    let textToCopy = "";
    if (item.input?.model_input?.type === "messages") {
      textToCopy = item.input.model_input.content
        .map((m) => `[${m.role}]\n${m.content}`)
        .join("\n\n");
    } else if (item.input?.model_input?.content) {
      textToCopy = item.input.model_input.content;
    } else if (item.input?.raw) {
      textToCopy = stringifyJson(item.input.raw);
    }
    const success = await copyToClipboard(textToCopy);
    if (success) showCopyFeedback(copyInputBtn);
  });

  if (item.has_output && item.output) {
    if (item.output.model_output) {
      outputFieldName.textContent = item.output.model_output.field_name || "";
      renderModelOutput(item.output, outputSection);
    } else {
      outputSection.innerHTML = '<div class="section-empty">No structured output</div>';
    }
  } else {
    outputSection.innerHTML = `
      <div class="pending-notice">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12,6 12,12 16,14"/>
        </svg>
        Output pending - waiting for model generation
      </div>
    `;
  }

  // Copy output button
  const copyOutputBtn = node.querySelector(".copy-output-btn");
  copyOutputBtn.addEventListener("click", async () => {
    let textToCopy = "";
    if (item.output?.model_output?.raw) {
      textToCopy = item.output.model_output.raw;
    } else if (item.output?.raw) {
      textToCopy = stringifyJson(item.output.raw);
    }
    const success = await copyToClipboard(textToCopy);
    if (success) showCopyFeedback(copyOutputBtn);
  });

  // Raw JSON
  const inputJsonBlock = node.querySelector(".input-json");
  const outputJsonBlock = node.querySelector(".output-json");
  inputJsonBlock.textContent = stringifyJson(item.input?.raw);
  outputJsonBlock.textContent = stringifyJson(item.output?.raw);

  // Raw JSON tab switching
  const rawTabs = node.querySelectorAll(".raw-tab");
  rawTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      rawTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const tabName = tab.dataset.tab;
      inputJsonBlock.classList.toggle("hidden", tabName !== "input");
      outputJsonBlock.classList.toggle("hidden", tabName !== "output");
    });
  });

  return node;
};

const showCopyFeedback = (btn) => {
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent-success)" stroke-width="2">
      <polyline points="20,6 9,17 4,12"/>
    </svg>
  `;
  setTimeout(() => {
    btn.innerHTML = originalHtml;
  }, 1500);
};

const renderStageDetail = (resp, cache) => {
  elements.stageDetail.innerHTML = "";

  // Summary
  const summary = document.createElement("div");
  summary.className = "stage-summary";
  summary.innerHTML = `
    <div><strong>R${resp.round_idx} ${resp.stage}</strong></div>
    <div class="stage-meta">
      Inputs: ${resp.input_total} · Outputs: ${resp.output_total} · Showing: ${cache.items.length}/${resp.total}
    </div>
  `;
  elements.stageDetail.appendChild(summary);

  // Items
  cache.items.forEach((item) => {
    const cardNode = renderStageItemCard(item, resp);
    elements.stageDetail.appendChild(cardNode);
  });

  // Load more button
  if (cache.offset < cache.total) {
    const btn = document.createElement("button");
    btn.className = "btn btn-secondary load-more";
    btn.textContent = `Load more (${cache.total - cache.offset} remaining)`;
    btn.addEventListener("click", () => {
      if (state.selected && cache.resp) {
        loadStage(state.selected, cache.resp.round_idx, cache.resp.stage, state.activeStageButton);
      }
    });
    elements.stageDetail.appendChild(btn);
  }
};

// Expand/Collapse all thinking sections
const expandAllThinking = () => {
  document.querySelectorAll(".thinking-section.collapsed").forEach((section) => {
    section.classList.remove("collapsed");
  });
};

const collapseAllThinking = () => {
  document.querySelectorAll(".thinking-section:not(.collapsed)").forEach((section) => {
    if (section.querySelector(".thinking-toggle")) {
      section.classList.add("collapsed");
    }
  });
};

// ============================================
// Load Problem Detail
// ============================================

const loadProblem = async (source, problemIdx) => {
  try {
    clearError();
    const data = await api(`/api/problem/${source}/${problemIdx}`);
    state.selected = data;

    // Update UI
    elements.detailTitle.textContent = `${data.source_name}/${data.problem_idx}`;
    elements.breadcrumbProblem.textContent = `${data.source_name}/${data.problem_idx}`;

    // Meta badges
    elements.detailProofCount.querySelector("span").textContent =
      `${data.proofs.length} proofs`;

    const bestScore = data.proofs.length
      ? Math.max(...data.proofs.map((p) => p.meanscore))
      : 0;
    elements.detailBestScore.querySelector("span").textContent =
      `Best: ${formatScore(bestScore)}`;

    // Problem text with math
    setMathContent(elements.problemText, data.question || "(No question found)");

    // Trend chart
    renderTrend(data.trend);

    // Per-problem score distribution
    renderScoreDistribution(buildDistributionFromProofs(data.proofs));

    // Proofs
    renderProofs(data.proofs);

    // Stage explorer
    renderStageExplorer(data);

    // Show detail view
    elements.detailEmpty.classList.add("hidden");
    elements.detailView.classList.remove("hidden");

    // Update problem list to show active state
    document.querySelectorAll(".problem-item").forEach((item) => {
      item.classList.toggle(
        "active",
        item.dataset.source === source && item.dataset.problem === problemIdx
      );
    });

    // Switch to proofs tab
    switchTab("proofs");
  } catch (err) {
    showError(`Problem load failed: ${err.message}`);
  }
};

// ============================================
// Bootstrap
// ============================================

const bootstrap = async () => {
  try {
    clearError();

    // Load summary
    const summary = await api("/api/summary");
    state.summary = summary;
    elements.runDir.textContent = summary.run_dir;
    renderTimeline(summary.stages);

    // Update header stats
    elements.statRounds.textContent = summary.rounds.length;

    // Load problems
    const problemsResp = await api("/api/problems");
    state.problems = problemsResp.problems;

    // Calculate total proofs
    const totalProofs = state.problems.reduce((sum, p) => sum + p.num_proofs, 0);
    elements.statProblems.textContent = state.problems.length;
    elements.statProofs.textContent = totalProofs;

    applyFiltersAndSort();

    if (state.selected) {
      loadProblem(state.selected.source_name, state.selected.problem_idx);
    }
  } catch (err) {
    showError(`Bootstrap failed: ${err.message}`);
  }
};

// ============================================
// Event Listeners
// ============================================

const initEventListeners = () => {
  // Problem list click
  elements.problemList.addEventListener("click", (event) => {
    const item = event.target.closest(".problem-item");
    if (!item) return;
    loadProblem(item.dataset.source, item.dataset.problem);
  });

  // Search
  elements.searchInput.addEventListener("input", () => {
    applyFiltersAndSort();
  });

  // Filter chips
  document.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.currentFilter = chip.dataset.filter;
      applyFiltersAndSort();
    });
  });

  // Proof sort
  elements.proofSortSelect.addEventListener("change", (e) => {
    state.currentSort = e.target.value;
    if (state.selected) {
      renderProofs(state.selected.proofs);
    }
  });

  // Tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      switchTab(tab.dataset.tab);
    });
  });

  // Expand/Collapse all thinking
  const expandAllBtn = document.getElementById("expandAllThinking");
  const collapseAllBtn = document.getElementById("collapseAllThinking");

  if (expandAllBtn) {
    expandAllBtn.addEventListener("click", expandAllThinking);
  }
  if (collapseAllBtn) {
    collapseAllBtn.addEventListener("click", collapseAllThinking);
  }

  // Refresh
  elements.refreshBtn.addEventListener("click", () => {
    bootstrap();
  });

  // Theme toggle
  elements.themeToggle.addEventListener("click", toggleTheme);

  // Copy problem
  elements.copyProblemBtn.addEventListener("click", async () => {
    if (state.selected?.question) {
      const success = await copyToClipboard(state.selected.question);
      if (success) {
        elements.copyProblemBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-success)" stroke-width="2">
            <polyline points="20,6 9,17 4,12"/>
          </svg>
        `;
        setTimeout(() => {
          elements.copyProblemBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
            </svg>
          `;
        }, 2000);
      }
    }
  });

  // Modal
  elements.closeModalBtn.addEventListener("click", closeProofModal);
  elements.proofModal.querySelector(".modal-backdrop").addEventListener("click", closeProofModal);

  elements.copyProofBtn.addEventListener("click", async () => {
    if (state.currentProofData?.proof) {
      const success = await copyToClipboard(state.currentProofData.proof);
      if (success) {
        const originalContent = elements.copyProofBtn.innerHTML;
        elements.copyProofBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-success)" stroke-width="2">
            <polyline points="20,6 9,17 4,12"/>
          </svg>
          Copied!
        `;
        setTimeout(() => {
          elements.copyProofBtn.innerHTML = originalContent;
        }, 2000);
      }
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !elements.proofModal.classList.contains("hidden")) {
      closeProofModal();
    }
  });
};

// ============================================
// Initialize
// ============================================

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initEventListeners();
  bootstrap();
});

// Also run immediately if DOM already loaded
if (document.readyState !== "loading") {
  initTheme();
  initEventListeners();
  bootstrap();
}
