/**
 * QuickSketch — app logic and Flask API integration.
 */

const API_BASE = 'http://localhost:5000';

const sketch = new SketchCanvas(document.getElementById('sketch-canvas'));

// DOM refs
const apiStatus = document.getElementById('api-status');
const statusText = apiStatus.querySelector('.status-text');
const categoryCount = document.getElementById('category-count');
const categoriesList = document.getElementById('categories-list');
const resultsEmpty = document.getElementById('results-empty');
const resultsLoading = document.getElementById('results-loading');
const resultsError = document.getElementById('results-error');
const resultsErrorText = document.getElementById('results-error-text');
const resultsList = document.getElementById('results-list');
const retryBtn = document.getElementById('retry-predict');
const canvasCard = document.querySelector('.canvas-card');

let categories = [];
let isPredicting = false;

// ── API helpers ──

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();
    setApiStatus(true, data.model_loaded ? 'Model ready' : 'API online');
    return data;
  } catch {
    setApiStatus(false, 'API offline');
    return null;
  }
}

async function loadCategories() {
  try {
    const res = await fetch(`${API_BASE}/categories`);
    if (!res.ok) throw new Error('Failed to load categories');
    const data = await res.json();
    categories = data.categories || [];
    renderCategories(categories);
    categoryCount.textContent = `${categories.length} categories supported`;
  } catch {
    categoryCount.textContent = 'Categories unavailable';
  }
}

async function predictSketch() {
  if (isPredicting) return;

  if (sketch.isEmpty()) {
    showError('Draw something on the canvas first.');
    return;
  }

  isPredicting = true;
  setPredictLoading(true);

  const imageBase64 = sketch.toBase64();

  console.log('Sending prediction request');
  console.log('Image base64 length:', imageBase64.length);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error('Invalid response from prediction service');
    }

    console.log('Prediction response:', data);

    if (!res.ok) {
      throw new Error(data.error || `Request failed (${res.status})`);
    }

    if (!data.predictions || !Array.isArray(data.predictions)) {
      throw new Error('Invalid prediction data received');
    }

    renderPredictions(data.predictions);
  } catch (err) {
    console.error('Prediction error:', err);

    const isNetworkError =
      err instanceof TypeError ||
      err.message === 'Failed to fetch' ||
      err.message.includes('NetworkError');

    showError(
      isNetworkError
        ? 'Prediction service unavailable'
        : err.message || 'Prediction failed. Is the backend running?'
    );
  } finally {
    isPredicting = false;
    setPredictLoading(false);
  }
}

// ── UI rendering ──

function setApiStatus(online, message) {
  apiStatus.classList.toggle('is-online', online);
  apiStatus.classList.toggle('is-offline', !online);
  statusText.textContent = message;
}

function renderCategories(list) {
  categoriesList.innerHTML = list
    .map((cat) => `<li class="category-chip">${escapeHtml(cat)}</li>`)
    .join('');
}

function renderPredictions(predictions) {
  hideAllResultStates();
  resultsList.classList.remove('is-hidden');
  resultsList.innerHTML = '';

  predictions.forEach((item, index) => {
    const li = document.createElement('li');
    li.className = 'result-card';
    li.innerHTML = `
      <span class="result-card__rank">${index === 0 ? 'Best match' : `#${index + 1}`}</span>
      <div class="result-card__row">
        <span class="result-card__name">${escapeHtml(item.category)}</span>
        <span class="result-card__confidence">${item.confidence.toFixed(1)}%</span>
      </div>
      <div class="result-card__bar" role="presentation">
        <div class="result-card__bar-fill" style="width: 0%"></div>
      </div>
    `;
    resultsList.appendChild(li);

    requestAnimationFrame(() => {
      li.querySelector('.result-card__bar-fill').style.width = `${Math.min(item.confidence, 100)}%`;
    });
  });
}

function showError(message) {
  hideAllResultStates();
  resultsError.classList.remove('is-hidden');
  resultsError.removeAttribute('aria-hidden');
  resultsErrorText.textContent = message;
}

function setPredictLoading(loading) {
  const predictBtn = document.querySelector('[data-action="predict"]');
  predictBtn.disabled = loading;
  canvasCard.classList.toggle('is-loading', loading);

  if (loading) {
    hideAllResultStates();
    resultsLoading.classList.remove('is-hidden');
    resultsLoading.removeAttribute('aria-hidden');
  }
}

function hideAllResultStates() {
  resultsEmpty.classList.add('is-hidden');
  resultsLoading.classList.add('is-hidden');
  resultsLoading.setAttribute('aria-hidden', 'true');
  resultsError.classList.add('is-hidden');
  resultsError.setAttribute('aria-hidden', 'true');
  resultsList.classList.add('is-hidden');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Tool rail events ──

document.querySelector('.tool-rail').addEventListener('click', (e) => {
  const btn = e.target.closest('.tool-btn');
  if (!btn) return;

  const tool = btn.dataset.tool;
  const action = btn.dataset.action;

  if (tool) {
    document.querySelectorAll('[data-tool]').forEach((b) => {
      b.classList.toggle('is-active', b === btn);
      b.setAttribute('aria-pressed', b === btn ? 'true' : 'false');
    });
    sketch.setTool(tool);
    return;
  }

  if (action === 'undo') sketch.undo();
  if (action === 'clear') {
    sketch.clear();
    hideAllResultStates();
    resultsEmpty.classList.remove('is-hidden');
  }
  if (action === 'download') sketch.download();
  if (action === 'predict') predictSketch();
});

retryBtn.addEventListener('click', predictSketch);

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    predictSketch();
  }
});

// ── Init ──

async function init() {
  await checkHealth();
  await loadCategories();
  setInterval(checkHealth, 30000);
}

init();
