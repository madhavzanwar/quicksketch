const ARTIFACT_BASES = [
  'model-artifacts',
  '../backend/models',
];

const ARTIFACT_PATHS = {
  summary: 'training_summary.json',
  history: 'training_history.json',
  confusionMatrix: 'confusion_matrix.json',
  classMetrics: 'class_metrics.json',
  confusionPairs: 'confusion_pairs.json',
  confusionMatrixImage: 'confusion_matrix.png',
};

const TRAINABLE_PARAMETERS_FALLBACK = 391952;
const TOP_CONFUSION_PAIRS_LIMIT = 6;

const summaryFields = {
  best_val_accuracy: document.querySelector('[data-summary="best_val_accuracy"]'),
  best_epoch: document.querySelector('[data-summary="best_epoch"]'),
  num_epochs_run: document.querySelector('[data-summary="num_epochs_run"]'),
  early_stopped: document.querySelector('[data-summary="early_stopped"]'),
  num_classes: document.querySelector('[data-summary="num_classes"]'),
  training_samples: document.querySelector('[data-summary="training_samples"]'),
  trainable_parameters: document.querySelector('[data-summary="trainable_parameters"]'),
};

const summaryNote = document.getElementById('summary-note');
const errorNote = document.getElementById('performance-error');
const metricsTableWrap = document.getElementById('metrics-table-wrap');
const metricsTableBody = document.getElementById('metrics-table-body');
const metricsFallback = document.getElementById('metrics-fallback');
const metricsNote = document.getElementById('metrics-note');
const confusionPairsGrid = document.getElementById('confusion-pairs-grid');
const pairsNote = document.getElementById('pairs-note');
const matrixFigure = document.getElementById('matrix-figure');
const matrixImage = document.getElementById('confusion-matrix-image');
const matrixCaption = document.getElementById('matrix-caption');
const matrixFallback = document.getElementById('matrix-fallback');
const matrixNote = document.getElementById('matrix-note');
const curveNote = document.getElementById('curve-note');
const heroMeta = document.getElementById('hero-meta');

function formatPercent(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(digits)}%` : 'Unavailable';
}

function formatBoolean(value) {
  if (typeof value !== 'boolean') return 'Unavailable';
  return value ? 'Yes' : 'No';
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : 'Unavailable';
}

function formatMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : 'N/A';
}

function formatCompactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 'Unavailable';
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(number);
}

function titleCase(value) {
  return String(value)
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function buildLinePath(points) {
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ');
}

function createEmptyState(container, message) {
  container.innerHTML = '';
  const empty = document.createElement('p');
  empty.className = 'chart-shell__empty';
  empty.textContent = message;
  container.appendChild(empty);
}

function setHeroMetaItems(items) {
  heroMeta.innerHTML = items
    .filter(Boolean)
    .map((item) => `<span class="hero-meta__item">${item}</span>`)
    .join('');
}

function showSectionError(message) {
  errorNote.textContent = message;
  errorNote.classList.remove('is-hidden');
}

function clearSectionError() {
  errorNote.classList.add('is-hidden');
}

async function fetchJson(url) {
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to load ${url}:`, error);
    return null;
  }
}

function buildArtifactUrlCandidates(fileName) {
  return ARTIFACT_BASES.map((base) => `${base}/${fileName}`);
}

async function fetchArtifactJson(fileName) {
  const candidates = buildArtifactUrlCandidates(fileName);
  for (const url of candidates) {
    const payload = await fetchJson(url);
    if (payload) {
      return {
        payload,
        sourceUrl: url,
      };
    }
  }

  return {
    payload: null,
    sourceUrl: null,
  };
}

function createChart(container, config) {
  const { title, trainData, valData, formatter } = config;

  if (!Array.isArray(trainData) || !Array.isArray(valData) || !trainData.length || !valData.length) {
    createEmptyState(container, 'Training curves unavailable.');
    return;
  }

  const width = 640;
  const height = 280;
  const padding = { top: 24, right: 24, bottom: 44, left: 52 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const epochs = trainData.map((_, index) => index + 1);
  const values = [...trainData, ...valData];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const yMin = min - range * 0.08;
  const yMax = max + range * 0.08;

  const toPoint = (value, index, source) => {
    const x = padding.left + (index / Math.max(source.length - 1, 1)) * chartWidth;
    const y = padding.top + ((yMax - value) / (yMax - yMin || 1)) * chartHeight;
    return { x, y };
  };

  const trainPoints = trainData.map((value, index) => toPoint(value, index, trainData));
  const valPoints = valData.map((value, index) => toPoint(value, index, valData));

  const yTicks = 4;
  const tickMarkup = Array.from({ length: yTicks + 1 }, (_, index) => {
    const value = yMin + ((yMax - yMin) * index) / yTicks;
    const y = padding.top + chartHeight - (chartHeight * index) / yTicks;
    return `
      <g class="chart-axis-group">
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="chart-grid-line"></line>
        <text x="${padding.left - 10}" y="${y + 4}" text-anchor="end" class="chart-axis-label">${formatter(value)}</text>
      </g>
    `;
  }).join('');

  const xTickMarkup = epochs.map((epoch, index) => {
    if (epochs.length > 8 && index % 2 !== 0 && index !== epochs.length - 1) {
      return '';
    }
    const x = padding.left + (index / Math.max(epochs.length - 1, 1)) * chartWidth;
    return `
      <g class="chart-axis-group">
        <line x1="${x}" y1="${padding.top + chartHeight}" x2="${x}" y2="${padding.top + chartHeight + 6}" class="chart-axis-tick"></line>
        <text x="${x}" y="${height - 14}" text-anchor="middle" class="chart-axis-label">${epoch}</text>
      </g>
    `;
  }).join('');

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="chart-svg" aria-label="${title}">
      <rect x="${padding.left}" y="${padding.top}" width="${chartWidth}" height="${chartHeight}" class="chart-plot"></rect>
      ${tickMarkup}
      <line x1="${padding.left}" y1="${padding.top + chartHeight}" x2="${width - padding.right}" y2="${padding.top + chartHeight}" class="chart-axis-line"></line>
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + chartHeight}" class="chart-axis-line"></line>
      ${xTickMarkup}
      <path d="${buildLinePath(trainPoints)}" class="chart-line chart-line--train"></path>
      <path d="${buildLinePath(valPoints)}" class="chart-line chart-line--val"></path>
      ${trainPoints.map((point, index) => `<circle cx="${point.x}" cy="${point.y}" r="3" class="chart-point chart-point--train"><title>Epoch ${index + 1}: ${formatter(trainData[index])}</title></circle>`).join('')}
      ${valPoints.map((point, index) => `<circle cx="${point.x}" cy="${point.y}" r="3" class="chart-point chart-point--val"><title>Epoch ${index + 1}: ${formatter(valData[index])}</title></circle>`).join('')}
    </svg>
  `;
}

function renderSummary(summary, confusionMatrix) {
  const classSummary = confusionMatrix?.class_summary || {};
  const validationSplit = confusionMatrix?.validation_split || {};
  const trainingSamples = Object.values(classSummary).reduce((total, entry) => {
    return total + Number(entry?.train || 0);
  }, 0);
  const validationSamples = Number(confusionMatrix?.total_samples || 0);
  const numClasses = Number(confusionMatrix?.num_classes || 0);

  summaryFields.best_val_accuracy.textContent = formatPercent(summary?.best_val_accuracy);
  summaryFields.best_epoch.textContent = summary?.best_epoch ?? 'Unavailable';
  summaryFields.num_epochs_run.textContent = summary?.num_epochs_run ?? 'Unavailable';
  summaryFields.early_stopped.textContent = formatBoolean(summary?.early_stopped);
  summaryFields.num_classes.textContent = formatNumber(numClasses);
  summaryFields.training_samples.textContent = formatNumber(trainingSamples);
  summaryFields.trainable_parameters.textContent = formatNumber(TRAINABLE_PARAMETERS_FALLBACK);

  setHeroMetaItems([
    numClasses ? `${formatNumber(numClasses)} classes` : null,
    trainingSamples ? `${formatNumber(trainingSamples)} training samples` : null,
    summary?.best_val_accuracy !== undefined ? `${formatPercent(summary.best_val_accuracy)} validation accuracy` : null,
  ]);

  if (summary || confusionMatrix) {
    const notes = [];
    if (summary?.random_seed !== undefined) {
      notes.push(`Random seed ${summary.random_seed}`);
    }
    if (trainingSamples) {
      notes.push(`${formatNumber(trainingSamples)} training samples inferred from the saved evaluation split`);
    }
    if (validationSamples) {
      notes.push(`${formatNumber(validationSamples)} held-out validation samples`);
    }
    if (validationSplit.train_ratio !== undefined) {
      notes.push(`${formatPercent(validationSplit.train_ratio * 100, 0)} / ${formatPercent((validationSplit.validation_ratio || 0) * 100, 0)} train-validation split`);
    }
    notes.push(`Trainable parameters shown for the current released checkpoint (${formatNumber(TRAINABLE_PARAMETERS_FALLBACK)})`);
    summaryNote.textContent = notes.join('. ') + '.';
  } else {
    summaryNote.textContent = 'Summary artifacts unavailable.';
    setHeroMetaItems(['Saved artifacts unavailable']);
  }
}

function renderCharts(history) {
  const lossAvailable = Array.isArray(history?.train_loss) && Array.isArray(history?.val_loss);
  const accAvailable = Array.isArray(history?.train_acc) && Array.isArray(history?.val_acc);

  createChart(document.getElementById('loss-chart'), {
    title: 'Loss over epochs',
    trainData: history?.train_loss,
    valData: history?.val_loss,
    formatter: (value) => Number(value).toFixed(2),
  });

  createChart(document.getElementById('accuracy-chart'), {
    title: 'Accuracy over epochs',
    trainData: history?.train_acc,
    valData: history?.val_acc,
    formatter: (value) => `${Number(value).toFixed(1)}%`,
  });

  if (lossAvailable && accAvailable) {
    const finalEpoch = history.train_loss.length;
    const finalValAcc = history.val_acc[history.val_acc.length - 1];
    const finalValLoss = history.val_loss[history.val_loss.length - 1];
    curveNote.textContent = `Rendered from ${formatNumber(finalEpoch)} saved epochs. Final validation loss: ${Number(finalValLoss).toFixed(3)}. Final validation accuracy: ${formatPercent(finalValAcc)}.`;
    curveNote.classList.remove('is-hidden');
  } else {
    curveNote.classList.add('is-hidden');
  }
}

function renderMetricsTable(metricsPayload) {
  const metrics = metricsPayload?.metrics;
  if (!Array.isArray(metrics) || !metrics.length) {
    metricsTableWrap.classList.add('is-hidden');
    metricsFallback.classList.remove('is-hidden');
    metricsNote.classList.add('is-hidden');
    return;
  }

  const macroF1 = metrics.reduce((total, metric) => total + Number(metric.f1_score || 0), 0) / metrics.length;
  const macroRecall = metrics.reduce((total, metric) => total + Number(metric.recall || 0), 0) / metrics.length;

  metricsFallback.classList.add('is-hidden');
  metricsTableWrap.classList.remove('is-hidden');
  metricsTableBody.innerHTML = metrics.map((metric) => `
    <tr>
      <th scope="row">${titleCase(metric.class_name)}</th>
      <td>${formatMetric(metric.precision)}</td>
      <td>${formatMetric(metric.recall)}</td>
      <td>${formatMetric(metric.f1_score)}</td>
      <td>${formatNumber(metric.support)}</td>
    </tr>
  `).join('');
  metricsNote.textContent = `Class order matches training and inference. Macro recall: ${formatMetric(macroRecall)}. Macro F1-score: ${formatMetric(macroF1)}.`;
  metricsNote.classList.remove('is-hidden');
}

function renderConfusionPairs(pairsPayload) {
  const pairs = pairsPayload?.pairs;
  if (!Array.isArray(pairs) || !pairs.length) {
    confusionPairsGrid.innerHTML = `
      <article class="insight-card insight-card--placeholder">
        <p class="insight-card__eyebrow">Unavailable</p>
        <h3 class="insight-card__title">Confusion insights missing</h3>
        <p class="insight-card__body">Most-confused class pairs could not be loaded from the saved artifact.</p>
      </article>
    `;
    pairsNote.classList.add('is-hidden');
    return;
  }

  confusionPairsGrid.innerHTML = pairs
    .slice(0, TOP_CONFUSION_PAIRS_LIMIT)
    .map((pair, index) => `
      <article class="insight-card">
        <p class="insight-card__eyebrow">Pair ${index + 1}</p>
        <h3 class="insight-card__title">${titleCase(pair.true_class)} -> ${titleCase(pair.predicted_class)}</h3>
        <p class="insight-card__body">
          ${formatNumber(pair.count)} validation sketches labeled as ${titleCase(pair.true_class)}
          were predicted as ${titleCase(pair.predicted_class)}.
        </p>
      </article>
    `)
    .join('');

  const topPair = pairs[0];
  pairsNote.classList.remove('is-hidden');
  pairsNote.textContent = `Top saved confusion: ${titleCase(topPair.true_class)} -> ${titleCase(topPair.predicted_class)} (${formatNumber(topPair.count)} sketches).`;
}

function setupConfusionMatrixImage(confusionMatrix) {
  if (!confusionMatrix) {
    matrixFigure.classList.add('is-hidden');
    matrixFallback.classList.remove('is-hidden');
    matrixNote.classList.add('is-hidden');
    return;
  }

  if (confusionMatrix?.accuracy !== undefined) {
    matrixCaption.textContent = `Held-out validation accuracy: ${formatPercent(confusionMatrix.accuracy * 100)}. Darker diagonal cells indicate stronger class separation.`;
  }
  if (confusionMatrix?.total_samples !== undefined) {
    matrixNote.textContent = `Image generated from ${formatNumber(confusionMatrix.total_samples)} validation predictions across ${formatNumber(confusionMatrix.num_classes)} classes.`;
  }

  const candidates = buildArtifactUrlCandidates(ARTIFACT_PATHS.confusionMatrixImage);

  const tryLoadImage = (index) => {
    if (index >= candidates.length) {
      matrixFigure.classList.add('is-hidden');
      matrixFallback.classList.remove('is-hidden');
      matrixNote.classList.add('is-hidden');
      return;
    }

    const nextUrl = candidates[index];
    matrixImage.onload = () => {
      matrixFigure.classList.remove('is-hidden');
      matrixFallback.classList.add('is-hidden');
      matrixNote.classList.remove('is-hidden');
    };
    matrixImage.onerror = () => {
      tryLoadImage(index + 1);
    };
    matrixImage.src = nextUrl;
  };

  tryLoadImage(0);
}

async function loadPerformance() {
  const [
    summary,
    history,
    confusionMatrix,
    classMetrics,
    confusionPairs,
  ] = await Promise.all([
    fetchArtifactJson(ARTIFACT_PATHS.summary),
    fetchArtifactJson(ARTIFACT_PATHS.history),
    fetchArtifactJson(ARTIFACT_PATHS.confusionMatrix),
    fetchArtifactJson(ARTIFACT_PATHS.classMetrics),
    fetchArtifactJson(ARTIFACT_PATHS.confusionPairs),
  ]);

  clearSectionError();

  renderSummary(summary.payload, confusionMatrix.payload);
  renderCharts(history.payload);
  renderMetricsTable(classMetrics.payload);
  renderConfusionPairs(confusionPairs.payload);
  setupConfusionMatrixImage(confusionMatrix.payload);

  const missingSections = [];
  if (!summary.payload || !confusionMatrix.payload) missingSections.push('summary');
  if (!history.payload) missingSections.push('training curves');
  if (!classMetrics.payload) missingSections.push('class metrics');
  if (!confusionPairs.payload) missingSections.push('confusion insights');

  if (missingSections.length) {
    showSectionError(`Some performance artifacts are missing or unavailable: ${missingSections.join(', ')}.`);
  }
}

loadPerformance();
