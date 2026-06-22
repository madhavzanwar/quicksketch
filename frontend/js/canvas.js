/**
 * SketchCanvas — drawing surface with pen, eraser, undo, and export.
 * Black strokes on white background (matches backend preprocessing).
 */
class SketchCanvas {
  constructor(canvasEl, options = {}) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.penSize = options.penSize ?? 4;
    this.eraserSize = options.eraserSize ?? 24;
    this.strokeColor = options.strokeColor ?? '#1A1A1A';
    this.tool = 'pen';
    this.isDrawing = false;
    this.history = [];
    this.maxHistory = 40;

    this._bindEvents();
    this._resize();

    // Resize when the card layout changes (not only on window resize)
    const observer = new ResizeObserver(() => this._resize());
    observer.observe(this.canvas.parentElement);
  }

  setTool(tool) {
    this.tool = tool;
    this.canvas.closest('.canvas-card')?.classList.toggle('is-eraser', tool === 'eraser');
  }

  clear() {
    const { width, height } = this.canvas;
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.fillRect(0, 0, width, height);
    this._pushHistory();
  }

  undo() {
    if (this.history.length <= 1) return false;
    this.history.pop();
    const snapshot = this.history[this.history.length - 1];
    this.ctx.putImageData(snapshot, 0, 0);
    return true;
  }

  canUndo() {
    return this.history.length > 1;
  }

  isEmpty() {
    const { width, height } = this.canvas;
    const data = this.ctx.getImageData(0, 0, width, height).data;
    // Any pixel that isn't pure white means something was drawn
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] !== 255 || data[i + 1] !== 255 || data[i + 2] !== 255) {
        return false;
      }
    }
    return true;
  }

  /** Returns base64 PNG string (with data URL prefix) */
  toDataURL() {
    return this.canvas.toDataURL('image/png');
  }

  /** Returns raw base64 without the data:image/png;base64, prefix */
  toBase64() {
    return this.toDataURL().split(',')[1];
  }

  download(filename = 'quicksketch.png') {
    const link = document.createElement('a');
    link.download = filename;
    link.href = this.toDataURL();
    link.click();
  }

  // ── Private ──

  _resize() {
    const card = this.canvas.parentElement;
    const rect = card.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);

    if (w === 0 || h === 0) return;

    const hadContent = this.canvas.width > 0;
    const backup = document.createElement('canvas');
    if (hadContent) {
      backup.width = this.canvas.width;
      backup.height = this.canvas.height;
      backup.getContext('2d').drawImage(this.canvas, 0, 0);
    }

    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.fillRect(0, 0, w, h);

    if (hadContent) {
      this.ctx.drawImage(backup, 0, 0, w, h);
    }

    if (this.history.length === 0) {
      this._pushHistory();
    }
  }

  _pushHistory() {
    const snapshot = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    this.history.push(snapshot);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
  }

  _getPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  _startStroke(e) {
    e.preventDefault();
    this.isDrawing = true;
    const point = this._getPoint(e);

    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    if (this.tool === 'eraser') {
      this.ctx.globalCompositeOperation = 'destination-out';
      this.ctx.lineWidth = this.eraserSize;
    } else {
      this.ctx.globalCompositeOperation = 'source-over';
      this.ctx.strokeStyle = this.strokeColor;
      this.ctx.lineWidth = this.penSize;
    }

    this.ctx.beginPath();
    this.ctx.moveTo(point.x, point.y);
  }

  _drawStroke(e) {
    if (!this.isDrawing) return;
    e.preventDefault();
    const point = this._getPoint(e);
    this.ctx.lineTo(point.x, point.y);
    this.ctx.stroke();
  }

  _endStroke(e) {
    if (!this.isDrawing) return;
    if (e) e.preventDefault();
    this.isDrawing = false;
    this.ctx.closePath();
    this.ctx.globalCompositeOperation = 'source-over';
    this._pushHistory();
  }

  _bindEvents() {
    const c = this.canvas;

    c.addEventListener('mousedown', (e) => this._startStroke(e));
    c.addEventListener('mousemove', (e) => this._drawStroke(e));
    c.addEventListener('mouseup', (e) => this._endStroke(e));
    c.addEventListener('mouseleave', (e) => this._endStroke(e));

    c.addEventListener('touchstart', (e) => this._startStroke(e), { passive: false });
    c.addEventListener('touchmove', (e) => this._drawStroke(e), { passive: false });
    c.addEventListener('touchend', (e) => this._endStroke(e));
    c.addEventListener('touchcancel', (e) => this._endStroke(e));
  }
}
