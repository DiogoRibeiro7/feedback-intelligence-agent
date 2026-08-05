import "./styles.css";
import {
  ApiError,
  postQuery,
  streamQuery,
  type AgentAnswer,
  type Citation,
  type StreamMetadata,
} from "./api";

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`Missing element: #${id}`);
  }
  return node as T;
}

const form = el<HTMLFormElement>("query-form");
const questionInput = el<HTMLTextAreaElement>("question");
const streamToggle = el<HTMLInputElement>("stream-toggle");
const submitBtn = el<HTMLButtonElement>("submit-btn");
const statusBox = el<HTMLElement>("status");
const dashboardPanel = el<HTMLElement>("dashboard-panel");
const dashboardSummary = el<HTMLElement>("dashboard-summary");
const latencyChart = el<HTMLElement>("latency-chart");
const scoreChart = el<HTMLElement>("score-chart");
const coverageChart = el<HTMLElement>("coverage-chart");
const answerPanel = el<HTMLElement>("answer-panel");
const answerBox = el<HTMLElement>("answer");
const actionsBox = el<HTMLElement>("actions");
const metaPanel = el<HTMLElement>("meta-panel");
const metaList = el<HTMLDListElement>("meta");
const sourcesPanel = el<HTMLElement>("sources-panel");
const sourcesList = el<HTMLOListElement>("sources");

function setStatus(message: string, kind: "info" | "error" | "none"): void {
  if (kind === "none") {
    statusBox.hidden = true;
    statusBox.textContent = "";
    return;
  }
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.className = `status status--${kind}`;
}

function clearResults(): void {
  answerPanel.hidden = true;
  metaPanel.hidden = true;
  sourcesPanel.hidden = true;
  answerBox.textContent = "";
  actionsBox.textContent = "";
  metaList.textContent = "";
  sourcesList.textContent = "";
}

function renderActions(actions: string[]): void {
  actionsBox.textContent = "";
  if (actions.length === 0) {
    return;
  }
  const heading = document.createElement("h3");
  heading.className = "actions__title";
  heading.textContent = "Recommended actions";
  const list = document.createElement("ul");
  for (const action of actions) {
    const item = document.createElement("li");
    item.textContent = action;
    list.appendChild(item);
  }
  actionsBox.appendChild(heading);
  actionsBox.appendChild(list);
}

function addMeta(term: string, value: string): void {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  metaList.appendChild(dt);
  metaList.appendChild(dd);
}

interface MetaView {
  provider?: string;
  latencyMs?: number;
  route: string;
  confidence: number;
}

interface DashboardSample {
  question: string;
  latencyMs: number;
  retrievalScores: number[];
  citationCoverage: number;
  citationCount: number;
}

const dashboardSamples: DashboardSample[] = [];

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function percentile(values: number[], percentileRank: number): number {
  if (values.length === 0) {
    return 0;
  }
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.ceil((percentileRank / 100) * ordered.length) - 1;
  return ordered[Math.max(0, Math.min(index, ordered.length - 1))];
}

function answerCitationMarkers(answer: string): Set<number> {
  const markers = new Set<number>();
  const matches = answer.matchAll(/\[(\d+)\]/g);
  for (const match of matches) {
    markers.add(Number(match[1]));
  }
  return markers;
}

function citationCoverage(answer: string, citations: Citation[]): number {
  const markers = answerCitationMarkers(answer);
  if (markers.size === 0) {
    return citations.length > 0 ? 0 : 1;
  }
  const citationIds = new Set(citations.map((citation) => citation.citation_id));
  let resolved = 0;
  for (const marker of markers) {
    if (citationIds.has(marker)) {
      resolved += 1;
    }
  }
  return resolved / markers.size;
}

function pushDashboardSample(
  question: string,
  answer: string,
  citations: Citation[],
  latencyMs: number,
): void {
  dashboardSamples.push({
    question,
    latencyMs,
    retrievalScores: citations.map((citation) => citation.score),
    citationCoverage: citationCoverage(answer, citations),
    citationCount: citations.length,
  });
  renderDashboard();
}

function renderSummaryMetric(label: string, value: string): HTMLElement {
  const item = document.createElement("div");
  item.className = "dashboard__metric";
  const valueNode = document.createElement("strong");
  valueNode.textContent = value;
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  item.appendChild(valueNode);
  item.appendChild(labelNode);
  return item;
}

function renderDashboard(): void {
  if (dashboardSamples.length === 0) {
    dashboardPanel.hidden = true;
    return;
  }
  const latencies = dashboardSamples.map((sample) => sample.latencyMs);
  const scores = dashboardSamples.flatMap((sample) => sample.retrievalScores);
  const averageCoverage = mean(dashboardSamples.map((sample) => sample.citationCoverage));
  dashboardSummary.textContent = "";
  dashboardSummary.appendChild(renderSummaryMetric("Queries", String(dashboardSamples.length)));
  dashboardSummary.appendChild(renderSummaryMetric("Avg latency", `${mean(latencies).toFixed(1)} ms`));
  dashboardSummary.appendChild(renderSummaryMetric("P95 latency", `${percentile(latencies, 95).toFixed(1)} ms`));
  dashboardSummary.appendChild(renderSummaryMetric("Avg score", scores.length ? mean(scores).toFixed(3) : "0.000"));
  dashboardSummary.appendChild(
    renderSummaryMetric("Citation coverage", `${Math.round(averageCoverage * 100)}%`),
  );
  renderLatencyChart(latencies);
  renderScoreChart(scores);
  renderCoverageChart();
  dashboardPanel.hidden = false;
}

function renderLatencyChart(latencies: number[]): void {
  latencyChart.textContent = "";
  const maxLatency = Math.max(...latencies, 1);
  for (const [index, latency] of latencies.slice(-8).entries()) {
    const row = document.createElement("div");
    row.className = "bar-chart__row";
    const label = document.createElement("span");
    label.textContent = `Q${dashboardSamples.length - latencies.slice(-8).length + index + 1}`;
    const bar = document.createElement("span");
    bar.className = "bar-chart__bar";
    bar.style.width = `${Math.max((latency / maxLatency) * 100, 4)}%`;
    const value = document.createElement("span");
    value.className = "bar-chart__value";
    value.textContent = `${latency.toFixed(0)} ms`;
    row.appendChild(label);
    row.appendChild(bar);
    row.appendChild(value);
    latencyChart.appendChild(row);
  }
}

function renderScoreChart(scores: number[]): void {
  scoreChart.textContent = "";
  const buckets = [
    { label: "High", count: scores.filter((score) => score >= 0.65).length },
    { label: "Medium", count: scores.filter((score) => score >= 0.35 && score < 0.65).length },
    { label: "Low", count: scores.filter((score) => score < 0.35).length },
  ];
  const maxCount = Math.max(...buckets.map((bucket) => bucket.count), 1);
  for (const bucket of buckets) {
    const row = document.createElement("div");
    row.className = "score-chart__row";
    const label = document.createElement("span");
    label.textContent = bucket.label;
    const track = document.createElement("span");
    track.className = "score-chart__track";
    const fill = document.createElement("span");
    fill.className = "score-chart__fill";
    fill.style.width = `${Math.max((bucket.count / maxCount) * 100, bucket.count ? 8 : 0)}%`;
    const value = document.createElement("span");
    value.textContent = String(bucket.count);
    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    scoreChart.appendChild(row);
  }
}

function renderCoverageChart(): void {
  coverageChart.textContent = "";
  for (const [index, sample] of dashboardSamples.slice(-8).entries()) {
    const item = document.createElement("div");
    item.className = "coverage-chart__item";
    const ring = document.createElement("span");
    ring.className = "coverage-chart__ring";
    ring.style.setProperty("--coverage", `${sample.citationCoverage * 360}deg`);
    ring.textContent = `${Math.round(sample.citationCoverage * 100)}%`;
    const label = document.createElement("span");
    label.textContent = `Q${dashboardSamples.length - dashboardSamples.slice(-8).length + index + 1}`;
    item.title = `${sample.question} · ${sample.citationCount} citations`;
    item.appendChild(ring);
    item.appendChild(label);
    coverageChart.appendChild(item);
  }
}

function renderMeta(view: MetaView): void {
  metaList.textContent = "";
  if (view.provider !== undefined) {
    addMeta("Provider", view.provider);
  }
  if (view.latencyMs !== undefined) {
    addMeta("Latency", `${view.latencyMs.toFixed(1)} ms`);
  }
  addMeta("Route", view.route);
  addMeta("Confidence", view.confidence.toFixed(3));
  metaPanel.hidden = false;
}

function renderSources(citations: Citation[]): void {
  sourcesList.textContent = "";
  if (citations.length === 0) {
    sourcesPanel.hidden = true;
    return;
  }
  for (const citation of citations) {
    const item = document.createElement("li");
    item.className = "source";

    const head = document.createElement("div");
    head.className = "source__head";
    const id = document.createElement("span");
    id.className = "source__id";
    id.textContent = `[${citation.citation_id}] ${citation.document_id}`;
    const tag = document.createElement("span");
    tag.className = "source__tag";
    tag.textContent = `${citation.source} · score ${citation.score.toFixed(3)}`;
    head.appendChild(id);
    head.appendChild(tag);

    const quote = document.createElement("blockquote");
    quote.className = "source__quote";
    quote.textContent = citation.quote;

    item.appendChild(head);
    item.appendChild(quote);
    sourcesList.appendChild(item);
  }
  sourcesPanel.hidden = false;
}

function renderAnswer(result: AgentAnswer): void {
  answerBox.textContent = result.answer;
  answerPanel.hidden = false;
  renderActions(result.recommended_actions);
  renderMeta({
    route: result.route,
    confidence: result.confidence,
  });
  renderSources(result.citations);
}

function setBusy(busy: boolean): void {
  submitBtn.disabled = busy;
  submitBtn.textContent = busy ? "Working…" : "Ask";
}

async function runNonStreaming(question: string): Promise<void> {
  setStatus("Querying…", "info");
  const started = performance.now();
  const response = await postQuery({ question });
  const latencyMs = performance.now() - started;
  renderAnswer(response.result);
  pushDashboardSample(question, response.result.answer, response.result.citations, latencyMs);
  setStatus("", "none");
}

async function runStreaming(question: string): Promise<void> {
  setStatus("Streaming…", "info");
  answerPanel.hidden = false;
  answerBox.textContent = "";
  let streamed = "";
  const started = performance.now();
  let metadataReceived = false;
  await streamQuery(
    { question },
    {
      onContent: (text: string) => {
        streamed += text;
        answerBox.textContent = streamed;
      },
      onMetadata: (metadata: StreamMetadata) => {
        renderActions(metadata.recommended_actions);
        renderMeta({
          provider: metadata.provider,
          latencyMs: metadata.latency_ms,
          route: metadata.route,
          confidence: metadata.confidence,
        });
        renderSources(metadata.citations);
        metadataReceived = true;
        pushDashboardSample(question, streamed, metadata.citations, metadata.latency_ms);
      },
    },
  );
  if (streamed.length > 0 && !metadataReceived) {
    pushDashboardSample(question, streamed, [], performance.now() - started);
  }
  setStatus("", "none");
}

form.addEventListener("submit", (event: SubmitEvent) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question.length < 3) {
    setStatus("Please enter a question of at least 3 characters.", "error");
    return;
  }
  clearResults();
  setBusy(true);
  const run = streamToggle.checked ? runStreaming(question) : runNonStreaming(question);
  run
    .catch((error: unknown) => {
      const message = error instanceof ApiError ? error.message : String(error);
      setStatus(`Request failed: ${message}`, "error");
    })
    .finally(() => {
      setBusy(false);
    });
});
