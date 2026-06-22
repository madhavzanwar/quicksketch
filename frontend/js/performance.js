const SUMMARY_URL = '../backend/models/training_summary.json';
const HISTORY_URL = '../backend/models/training_history.json';
const METADATA_URL = 'http://localhost:5000/model-metadata';

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

function setLoadingError(message) {
  errorNote.textContent = message;
  errorNote.classList.remove('is-hidden');

  Object.values(summaryFields).forEach((node) => {
    node.textContent = 'Unavailable';
  });

  document.querySelectorAll('.chart-shell').forEach((shell) => {
    shell.innerHTML = '';
    const empty = document.createElement('p');
    empty.className = 'chart-shell__empty';
    empty.textContent = 'Training curves unavailable.';
    shell.appendChild(empty);
  });
}

function formatPercent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function formatBoolean(value) {
  return value ? 'Yes' : 'No';
}

function formatNumber(value) {
  return Number(value).toLocaleString();
}

function buildLinePath(points) {
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
}

function createChart(container, config) {
  const { title, trainData, valData, formatter } = config;
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

function renderSummary(summary) {
  summaryFields.best_val_accuracy.textContent = formatPercent(summary.best_val_accuracy);
  summaryFields.best_epoch.textContent = summary.best_epoch;
  summaryFields.num_epochs_run.textContent = summary.num_epochs_run;
  summaryFields.early_stopped.textContent = formatBoolean(summary.early_stopped);
}

function renderMetadata(metadata, summary) {
  summaryFields.num_classes.textContent = formatNumber(metadata.num_classes);
  summaryFields.training_samples.textContent = formatNumber(metadata.training_samples);
  summaryFields.trainable_parameters.textContent = formatNumber(metadata.trainable_parameters);
  summaryNote.textContent = `Summary loaded from training artifacts and live model metadata with random seed ${summary.random_seed}.`;
}

function renderCharts(history) {
  createChart(document.getElementById('loss-chart'), {
    title: 'Loss over epochs',
    trainData: history.train_loss,
    valData: history.val_loss,
    formatter: (value) => value.toFixed(2),
  });

  createChart(document.getElementById('accuracy-chart'), {
    title: 'Accuracy over epochs',
    trainData: history.train_acc,
    valData: history.val_acc,
    formatter: (value) => `${value.toFixed(1)}%`,
  });
}

async function loadPerformance() {
  try {
    const [summaryRes, historyRes, metadataRes] = await Promise.all([
      fetch(SUMMARY_URL),
      fetch(HISTORY_URL),
      fetch(METADATA_URL),
    ]);

    if (!summaryRes.ok || !historyRes.ok || !metadataRes.ok) {
      throw new Error('Training artifacts are unavailable.');
    }

    const [summary, history, metadata] = await Promise.all([
      summaryRes.json(),
      historyRes.json(),
      metadataRes.json(),
    ]);

    renderSummary(summary);
    renderMetadata(metadata, summary);
    renderCharts(history);
  } catch (error) {
    console.error('Failed to load performance data:', error);
    setLoadingError('Unable to load training artifacts or model metadata. Make sure the project server and backend are both running, then try again.');
  }
}

loadPerformance();
