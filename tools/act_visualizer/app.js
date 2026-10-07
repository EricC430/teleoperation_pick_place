/**
 * ACT Policy Attention Loss & Layer Latent Visualizer - Core Application
 */

// State
const state = {
  data: null,
  currentFrame: 0,
  isPlaying: false,
  fps: 15,
  playTimer: null,
  attnMode: "t0",
  opacity: 0.55,
  isDrawingROI: false,
  activeCamROI: null, // "front" or "wrist"
  dragStart: null,
  roiKeyframes: {
    front: {}, // { frameIdx: { x, y, w, h } } (normalized 0..1)
    wrist: {},
  },
  interpolatedROI: {
    front: [],
    wrist: [],
  },
  aotrTimeline: {
    front: [],
    wrist: [],
  },
  cachedImages: {
    front: {},
    wrist: {},
  },
};

// Turbo Colormap lookup (256 RGB values)
const TURBO_COLORMAP = [
  [48,18,59],[50,22,67],[51,25,75],[53,29,83],[54,33,91],[55,36,99],[57,40,107],[58,44,115],
  [59,47,123],[60,51,131],[61,55,138],[62,58,146],[63,62,153],[64,66,160],[65,70,167],[66,74,174],
  [67,78,181],[68,82,187],[69,86,193],[70,90,199],[70,94,205],[70,98,211],[70,102,216],[70,106,222],
  [70,111,227],[70,115,232],[69,119,236],[69,123,241],[68,128,245],[67,132,249],[66,136,252],[65,140,255],
  [63,145,255],[62,149,255],[60,153,255],[58,157,254],[56,161,252],[54,165,250],[52,170,247],[50,174,244],
  [48,178,240],[46,182,236],[44,186,231],[42,190,226],[40,194,221],[38,198,215],[37,202,209],[36,206,203],
  [36,210,197],[36,214,190],[36,218,184],[38,222,177],[40,225,171],[43,229,164],[46,232,158],[50,236,151],
  [54,239,145],[59,242,139],[64,245,133],[70,248,127],[76,250,121],[83,252,116],[90,254,111],[98,255,106],
  [106,255,102],[114,255,98],[123,255,94],[131,255,91],[140,254,88],[149,253,86],[158,252,84],[167,250,83],
  [176,248,82],[185,246,81],[193,243,81],[202,241,81],[210,238,81],[218,235,82],[226,231,83],[233,228,84],
  [240,224,85],[246,220,86],[251,215,88],[255,210,89],[255,205,91],[255,199,93],[255,193,94],[255,187,96],
  [255,181,97],[255,174,99],[255,168,100],[255,161,101],[255,154,101],[255,147,101],[254,140,101],[253,132,100],
  [251,125,99],[249,117,97],[246,109,95],[243,101,92],[239,94,89],[235,86,85],[231,79,81],[226,71,77],
  [221,64,72],[215,57,67],[209,51,62],[203,45,56],[196,39,50],[189,34,44],[182,29,38],[175,25,32]
];

// DOM Elements
const elements = {
  selectAttnMode: document.getElementById("select-attn-mode"),
  sliderOpacity: document.getElementById("slider-opacity"),
  labelOpacity: document.getElementById("label-opacity"),
  btnToggleROI: document.getElementById("btn-toggle-roi"),
  btnInterpolateROI: document.getElementById("btn-interpolate-roi"),
  btnClearROI: document.getElementById("btn-clear-roi"),
  
  canvasFrontImg: document.getElementById("canvas-front-img"),
  canvasFrontOverlay: document.getElementById("canvas-front-overlay"),
  canvasFrontROI: document.getElementById("canvas-front-roi"),
  valAoTRFront: document.getElementById("val-aotr-front"),
  badgeROIFront: document.getElementById("badge-roi-front"),

  canvasWristImg: document.getElementById("canvas-wrist-img"),
  canvasWristOverlay: document.getElementById("canvas-wrist-overlay"),
  canvasWristROI: document.getElementById("canvas-wrist-roi"),
  valAoTRWrist: document.getElementById("val-aotr-wrist"),
  badgeROIWrist: document.getElementById("badge-roi-wrist"),

  btnPlayPause: document.getElementById("btn-play-pause"),
  btnStepPrev: document.getElementById("btn-step-prev"),
  btnStepNext: document.getElementById("btn-step-next"),
  timelineScrubber: document.getElementById("timeline-scrubber"),
  txtFrameCounter: document.getElementById("txt-frame-counter"),
  txtTimeCounter: document.getElementById("txt-time-counter"),
  selectFPS: document.getElementById("select-fps"),

  canvasTrajectories: document.getElementById("canvas-trajectories"),
  tbodyJoints: document.getElementById("tbody-joints"),
  cardAlertBox: document.getElementById("card-alert-box"),
  alertTitle: document.getElementById("alert-title"),
  alertDesc: document.getElementById("alert-desc"),

  canvasAoTRTimeline: document.getElementById("canvas-aotr-timeline"),
  valMeanAoTRFront: document.getElementById("val-mean-aotr-front"),
  valMeanAoTRWrist: document.getElementById("val-mean-aotr-wrist"),
  valLostFrameCount: document.getElementById("val-lost-frame-count"),
  btnJumpFirstLost: document.getElementById("btn-jump-first-lost"),

  canvasChunkRollout: document.getElementById("canvas-chunk-rollout"),
  tabButtons: document.querySelectorAll(".tab-btn"),
  tabPanes: document.querySelectorAll(".tab-pane"),

  badgeCheckpoint: document.getElementById("badge-checkpoint"),
  badgeDataset: document.getElementById("badge-dataset"),
  badgeEpisode: document.getElementById("badge-episode"),
  btnLoadDefault: document.getElementById("btn-load-default"),
  dataFileInput: document.getElementById("data-file-input"),
};

// Initialize
function init() {
  bindEvents();
  // Try loading default bundle if available
  loadDefaultData();
}

function bindEvents() {
  elements.btnPlayPause.addEventListener("click", togglePlay);
  elements.btnStepPrev.addEventListener("click", stepPrev);
  elements.btnStepNext.addEventListener("click", stepNext);

  elements.timelineScrubber.addEventListener("input", (e) => {
    seekFrame(parseInt(e.target.value, 10));
  });

  elements.selectFPS.addEventListener("change", (e) => {
    state.fps = parseInt(e.target.value, 10);
    if (state.isPlaying) {
      stopPlayback();
      startPlayback();
    }
  });

  elements.selectAttnMode.addEventListener("change", (e) => {
    state.attnMode = e.target.value;
    renderCurrentFrame();
  });

  elements.sliderOpacity.addEventListener("input", (e) => {
    state.opacity = parseInt(e.target.value, 10) / 100.0;
    elements.labelOpacity.textContent = `${e.target.value}%`;
    renderCurrentFrame();
  });

  elements.btnToggleROI.addEventListener("click", () => {
    state.isDrawingROI = !state.isDrawingROI;
    elements.btnToggleROI.classList.toggle("active", state.isDrawingROI);
    elements.btnToggleROI.textContent = state.isDrawingROI ? "🛑 Exit ROI Draw" : "✏️ Draw ROI (Target Box)";
  });

  elements.btnInterpolateROI.addEventListener("click", () => {
    interpolateAllROIs();
    computeFullEpisodeAoTR();
    renderCurrentFrame();
  });

  elements.btnClearROI.addEventListener("click", () => {
    state.roiKeyframes.front = {};
    state.roiKeyframes.wrist = {};
    state.interpolatedROI.front = [];
    state.interpolatedROI.wrist = [];
    state.aotrTimeline.front = [];
    state.aotrTimeline.wrist = [];
    elements.badgeROIFront.textContent = "No ROI Set";
    elements.badgeROIWrist.textContent = "No ROI Set";
    elements.badgeROIFront.classList.remove("active");
    elements.badgeROIWrist.classList.remove("active");
    computeFullEpisodeAoTR();
    renderCurrentFrame();
  });

  // File loading
  elements.btnLoadDefault.addEventListener("click", loadDefaultData);
  elements.dataFileInput.addEventListener("change", handleFileUpload);

  // Tab navigation
  elements.tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      elements.tabButtons.forEach(b => b.classList.remove("active"));
      elements.tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      document.getElementById(targetId).classList.add("active");
      if (targetId === "tab-aotr") drawAoTRTimelineChart();
      if (targetId === "tab-chunk") drawChunkRolloutChart();
    });
  });

  // Jump button
  elements.btnJumpFirstLost.addEventListener("click", () => {
    const values = state.aotrTimeline.front;
    const measured = values.filter(v => v !== null && v !== undefined);
    if (measured.length) seekFrame(values.indexOf(Math.min(...measured)));
  });

  // Canvas Mouse Events for ROI Bounding Box
  setupROICanvas(elements.canvasFrontROI, "front");
  setupROICanvas(elements.canvasWristROI, "wrist");

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      stepPrev();
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      stepNext();
    } else if (e.code === "Home") {
      seekFrame(0);
    } else if (e.code === "End" && state.data) {
      seekFrame(state.data.metadata.total_frames - 1);
    }
  });
}

function setupROICanvas(canvas, camKey) {
  let isDragging = false;
  let startX = 0, startY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (!state.isDrawingROI) return;
    const rect = canvas.getBoundingClientRect();
    startX = (e.clientX - rect.left) / rect.width;
    startY = (e.clientY - rect.top) / rect.height;
    isDragging = true;
    state.activeCamROI = camKey;
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging || !state.isDrawingROI) return;
    const rect = canvas.getBoundingClientRect();
    const currX = (e.clientX - rect.left) / rect.width;
    const currY = (e.clientY - rect.top) / rect.height;

    const x = Math.min(startX, currX);
    const y = Math.min(startY, currY);
    const w = Math.abs(currX - startX);
    const h = Math.abs(currY - startY);

    drawPreviewROI(canvas, x, y, w, h);
  });

  const finishDrag = (e) => {
    if (!isDragging) return;
    isDragging = false;
    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / rect.width;
    const endY = (e.clientY - rect.top) / rect.height;

    const x = Math.max(0, Math.min(startX, endX));
    const y = Math.max(0, Math.min(startY, endY));
    const w = Math.min(1.0 - x, Math.abs(endX - startX));
    const h = Math.min(1.0 - y, Math.abs(endY - startY));

    if (w > 0.02 && h > 0.02) {
      // Save keyframe ROI
      state.roiKeyframes[camKey][state.currentFrame] = { x, y, w, h };
      const badge = camKey === "front" ? elements.badgeROIFront : elements.badgeROIWrist;
      badge.textContent = `ROI Key (f${state.currentFrame})`;
      badge.classList.add("active");

      interpolateAllROIs();
      computeFullEpisodeAoTR();
      renderCurrentFrame();
    }
  };

  canvas.addEventListener("mouseup", finishDrag);
  canvas.addEventListener("mouseleave", finishDrag);
}

function drawPreviewROI(canvas, x, y, w, h) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(x * canvas.width, y * canvas.height, w * canvas.width, h * canvas.height);
  ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
  ctx.fillRect(x * canvas.width, y * canvas.height, w * canvas.width, h * canvas.height);
}

// Data loading
async function loadDefaultData() {
  elements.btnLoadDefault.textContent = "⏳ Loading...";
  try {
    // Attempt to load from relative output directory
    const configuredUrl = new URLSearchParams(window.location.search).get("data");
    const defaultUrl = configuredUrl || "../../outputs/act_analysis/phase_b1_uvc60_ep0/ep0_data.json";
    const res = await fetch(defaultUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const bundleUrl = new URL(defaultUrl, window.location.href);
    onDataLoaded(data, new URL(".", bundleUrl).href.replace(/\/$/, ""));
  } catch (err) {
    console.warn("Could not load default file via relative path:", err);
    elements.btnLoadDefault.textContent = "⚡ Load Phase B1 (Ep 0)";
    alert("Could not automatically load default data file. Please click '📂 Load Data JSON' to select ep0_data.json from outputs/act_analysis/!");
  }
}

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result);
      // Path base assuming relative to tools/act_visualizer/
      const basePath = data.metadata.bundle_path
        ? `../../${data.metadata.bundle_path}`
        : `../../outputs/act_analysis/${data.metadata.run_name}_ep${data.metadata.episode}`;
      onDataLoaded(data, basePath);
    } catch (err) {
      alert("Error parsing JSON file: " + err.message);
    }
  };
  reader.readAsText(file);
}

function onDataLoaded(data, basePath) {
  state.data = data;
  state.basePath = basePath;
  state.currentFrame = 0;
  state.cachedImages = {front: {}, wrist: {}};
  state.roiKeyframes = {front: {}, wrist: {}};
  state.interpolatedROI = {front: [], wrist: []};
  state.aotrTimeline = {front: [], wrist: []};
  elements.badgeROIFront.textContent = "No ROI Set";
  elements.badgeROIWrist.textContent = "No ROI Set";
  elements.badgeROIFront.classList.remove("active");
  elements.badgeROIWrist.classList.remove("active");
  elements.valMeanAoTRFront.textContent = "--";
  elements.valMeanAoTRWrist.textContent = "--";
  elements.valLostFrameCount.textContent = "0 frames";
  elements.btnJumpFirstLost.disabled = true;

  const totalFrames = data.metadata.total_frames;
  elements.timelineScrubber.max = totalFrames - 1;
  elements.timelineScrubber.value = 0;

  elements.badgeCheckpoint.textContent = `Model: ${data.metadata.run_name}`;
  elements.badgeDataset.textContent = `Dataset: ${data.metadata.dataset_repo_id}`;
  elements.badgeEpisode.textContent = `Ep: ${data.metadata.episode}`;
  elements.btnLoadDefault.textContent = `✅ Loaded Ep ${data.metadata.episode}`;

  // Preload first few images
  preloadImagesAround(0);

  renderCurrentFrame();
  drawTrajectoryChart();
}

function preloadImagesAround(centerFrame) {
  if (!state.data) return;
  const total = state.data.metadata.total_frames;
  const start = Math.max(0, centerFrame - 5);
  const end = Math.min(total - 1, centerFrame + 15);

  for (let f = start; f <= end; f++) {
    const padF = String(f).padStart(4, "0");
    if (!state.cachedImages.front[f]) {
      const imgFront = new Image();
      imgFront.src = `${state.basePath}/frames/front_left/frame_${padF}.jpg`;
      state.cachedImages.front[f] = imgFront;
    }
    if (!state.cachedImages.wrist[f]) {
      const imgWrist = new Image();
      imgWrist.src = `${state.basePath}/frames/wrist/frame_${padF}.jpg`;
      state.cachedImages.wrist[f] = imgWrist;
    }
  }
}

// Playback Control
function togglePlay() {
  if (state.isPlaying) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (!state.data) return;
  state.isPlaying = true;
  elements.btnPlayPause.textContent = "⏸";
  elements.btnPlayPause.classList.add("btn-danger");

  const intervalMs = 1000 / state.fps;
  state.playTimer = setInterval(() => {
    if (state.currentFrame < state.data.metadata.total_frames - 1) {
      seekFrame(state.currentFrame + 1);
    } else {
      seekFrame(0); // loop
    }
  }, intervalMs);
}

function stopPlayback() {
  state.isPlaying = false;
  elements.btnPlayPause.textContent = "▶";
  elements.btnPlayPause.classList.remove("btn-danger");
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

function stepPrev() {
  if (state.currentFrame > 0) seekFrame(state.currentFrame - 1);
}

function stepNext() {
  if (state.data && state.currentFrame < state.data.metadata.total_frames - 1) {
    seekFrame(state.currentFrame + 1);
  }
}

function seekFrame(frameIdx) {
  if (!state.data) return;
  state.currentFrame = Math.max(0, Math.min(state.data.metadata.total_frames - 1, frameIdx));
  elements.timelineScrubber.value = state.currentFrame;
  preloadImagesAround(state.currentFrame);
  renderCurrentFrame();
}

// Frame Rendering
function renderCurrentFrame() {
  if (!state.data) return;

  const fIdx = state.currentFrame;
  const totalFrames = state.data.metadata.total_frames;
  const frameInfo = state.data.frames[fIdx];
  const fps = state.data.metadata.fps || 15.0;

  elements.txtFrameCounter.textContent = `Frame ${fIdx} / ${totalFrames - 1}`;
  elements.txtTimeCounter.textContent = `(${(fIdx / fps).toFixed(2)}s)`;

  // 1. Draw Raw Camera Frames
  renderCameraImage(elements.canvasFrontImg, state.cachedImages.front[fIdx], fIdx, "front_left");
  renderCameraImage(elements.canvasWristImg, state.cachedImages.wrist[fIdx], fIdx, "wrist");

  // 2. Draw Heatmap Overlays
  renderHeatmap(elements.canvasFrontOverlay, frameInfo, "observation.images.front-left", state.attnMode);
  renderHeatmap(elements.canvasWristOverlay, frameInfo, "observation.images.wrist", state.attnMode);

  // 3. Draw Active ROI Boxes & Compute AoTR
  renderROI(elements.canvasFrontROI, "front", fIdx);
  renderROI(elements.canvasWristROI, "wrist", fIdx);

  // 4. Update Tables & Latent Cards
  updateJointTable(fIdx);
  updateLatentCards(frameInfo);
  updateDiagnosticStatus();

  // 5. Update timeline vertical cursor
  drawTrajectoryChart();
  drawAoTRTimelineChart();
  drawChunkRolloutChart();
}

function renderCameraImage(canvas, cachedImg, fIdx, camDir) {
  const ctx = canvas.getContext("2d");
  if (cachedImg && cachedImg.complete && cachedImg.naturalWidth > 0) {
    ctx.drawImage(cachedImg, 0, 0, canvas.width, canvas.height);
  } else {
    // Load directly
    const img = new Image();
    const padF = String(fIdx).padStart(4, "0");
    img.src = `${state.basePath}/frames/${camDir}/frame_${padF}.jpg`;
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      state.cachedImages[camDir === "front_left" ? "front" : "wrist"][fIdx] = img;
    };
  }
}

function renderHeatmap(canvas, frameInfo, camKey, mode) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.opacity <= 0.01) return;

  // Retrieve 2D grid matrix
  let map2d = null;
  if (mode.startsWith("layer_")) {
    const lData = frameInfo.layer_latents[mode];
    if (lData && lData.cam_norm_maps) {
      map2d = lData.cam_norm_maps[camKey];
    }
  } else {
    if (frameInfo.heatmaps && frameInfo.heatmaps[camKey]) {
      map2d = frameInfo.heatmaps[camKey][mode];
    }
  }

  if (!map2d || map2d.length === 0) return;

  const rows = map2d.length;
  const cols = map2d[0].length;

  // Create offscreen canvas matching grid resolution
  const offscreen = document.createElement("canvas");
  offscreen.width = cols;
  offscreen.height = rows;
  const offCtx = offscreen.getContext("2d");
  const imgData = offCtx.createImageData(cols, rows);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = Math.max(0, Math.min(1, map2d[r][c]));
      const colorIdx = Math.floor(val * (TURBO_COLORMAP.length - 1));
      const rgb = TURBO_COLORMAP[colorIdx];
      const pixelIdx = (r * cols + c) * 4;
      imgData.data[pixelIdx] = rgb[0];
      imgData.data[pixelIdx + 1] = rgb[1];
      imgData.data[pixelIdx + 2] = rgb[2];
      imgData.data[pixelIdx + 3] = Math.floor(val * 255 * state.opacity);
    }
  }

  offCtx.putImageData(imgData, 0, 0);

  // Smoothly upscale to camera resolution
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

// ROI Handling & AoTR Computation
function renderROI(canvas, camKey, fIdx) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const roi = state.interpolatedROI[camKey][fIdx];
  const aotrVal = state.aotrTimeline[camKey][fIdx];
  const aotrLabel = camKey === "front" ? elements.valAoTRFront : elements.valAoTRWrist;

  if (!roi) {
    aotrLabel.textContent = "--%";
    aotrLabel.classList.remove("lost");
    return;
  }

  const pxX = roi.x * canvas.width;
  const pxY = roi.y * canvas.height;
  const pxW = roi.w * canvas.width;
  const pxH = roi.h * canvas.height;

  // Draw ROI Box
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.strokeRect(pxX, pxY, pxW, pxH);

  ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
  ctx.fillRect(pxX, pxY, pxW, pxH);

  // Draw Keyframe indicator corner
  if (state.roiKeyframes[camKey][fIdx]) {
    ctx.fillStyle = "#eab308";
    ctx.beginPath();
    ctx.arc(pxX + 6, pxY + 6, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Label
  const aotrPct = aotrVal != null ? (aotrVal * 100).toFixed(1) : "--";
  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 12px JetBrains Mono, monospace";
  ctx.fillText(`Target (${aotrPct}%)`, pxX + 4, pxY - 6 > 14 ? pxY - 6 : pxY + 16);

  // Update Footer Readout
  aotrLabel.textContent = `${aotrPct}%`;
  aotrLabel.classList.remove("lost");
}

function interpolateAllROIs() {
  if (!state.data) return;
  const total = state.data.metadata.total_frames;

  ["front", "wrist"].forEach(camKey => {
    const keyframes = Object.keys(state.roiKeyframes[camKey])
      .map(k => parseInt(k, 10))
      .sort((a, b) => a - b);

    state.interpolatedROI[camKey] = new Array(total).fill(null);

    if (keyframes.length === 0) return;

    if (keyframes.length === 1) {
      const singleROI = state.roiKeyframes[camKey][keyframes[0]];
      state.interpolatedROI[camKey][keyframes[0]] = { ...singleROI };
      return;
    }

    // Linear interpolation between keyframes
    for (let k = 0; k < keyframes.length - 1; k++) {
      const f1 = keyframes[k];
      const f2 = keyframes[k + 1];
      const roi1 = state.roiKeyframes[camKey][f1];
      const roi2 = state.roiKeyframes[camKey][f2];

      for (let f = f1; f <= f2; f++) {
        const alpha = (f - f1) / (f2 - f1);
        state.interpolatedROI[camKey][f] = {
          x: roi1.x + alpha * (roi2.x - roi1.x),
          y: roi1.y + alpha * (roi2.y - roi1.y),
          w: roi1.w + alpha * (roi2.w - roi1.w),
          h: roi1.h + alpha * (roi2.h - roi1.h),
        };
      }
    }

    // Visibility is unknown outside the annotated interval. Do not extrapolate
    // a target box to frames where the object may not yet be in view.
  });
}

function computeFullEpisodeAoTR() {
  if (!state.data) return;
  const total = state.data.metadata.total_frames;

  ["front", "wrist"].forEach(camKey => {
    const fullCamKey = camKey === "front" ? "observation.images.front-left" : "observation.images.wrist";
    state.aotrTimeline[camKey] = new Array(total).fill(null);

    let sumAoTR = 0;
    let countAoTR = 0;
    const meanLabel = camKey === "front" ? elements.valMeanAoTRFront : elements.valMeanAoTRWrist;
    meanLabel.textContent = "--";

    for (let f = 0; f < total; f++) {
      const roi = state.interpolatedROI[camKey][f];
      if (!roi) continue;

      const fInfo = state.data.frames[f];
      if (!fInfo || !fInfo.heatmaps || !fInfo.heatmaps[fullCamKey]) continue;

      // Per-camera display heatmaps have had their minimum subtracted; their
      // sums are not attention mass. Older bundles must be re-exported.
      const map2d = fInfo.heatmaps_raw?.[fullCamKey]?.["t0"];
      if (!map2d) continue;

      const rows = map2d.length;
      const cols = map2d[0].length;

      const colStart = Math.max(0, Math.floor(roi.x * cols));
      const colEnd = Math.min(cols, Math.ceil((roi.x + roi.w) * cols));
      const rowStart = Math.max(0, Math.floor(roi.y * rows));
      const rowEnd = Math.min(rows, Math.ceil((roi.y + roi.h) * rows));

      let roiSum = 0;
      let totalSum = 0;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = map2d[r][c];
          totalSum += val;
          if (r >= rowStart && r < rowEnd && c >= colStart && c < colEnd) {
            roiSum += val;
          }
        }
      }

      if (totalSum <= 1e-12) continue;
      const aotr = roiSum / totalSum;
      state.aotrTimeline[camKey][f] = aotr;
      sumAoTR += aotr;
      countAoTR++;
    }

    if (countAoTR > 0) {
      const meanVal = ((sumAoTR / countAoTR) * 100).toFixed(1) + "%";
      meanLabel.textContent = meanVal;
    }
    if (camKey === "wrist") elements.valLostFrameCount.textContent = `${countAoTR} frames`;
    if (camKey === "front") elements.btnJumpFirstLost.disabled = countAoTR === 0;
  });

  drawAoTRTimelineChart();
}

function updateDiagnosticStatus() {
  const f = state.currentFrame;
  const aotrFront = state.aotrTimeline.front[f];
  const aotrWrist = state.aotrTimeline.wrist[f];

  if (aotrFront == null && aotrWrist == null) {
    elements.cardAlertBox.className = "diagnosis-card";
    elements.alertTitle.textContent = "Visual Status: Monitoring";
    elements.alertDesc.textContent = "Draw an ROI on visible objects. ROI measurements require a bundle with raw attention weights; older bundles need re-exporting.";
    return;
  }

  elements.cardAlertBox.className = "diagnosis-card";
  elements.alertTitle.textContent = "ROI attention mass (descriptive)";
  const pct = value => value == null ? "--" : `${(value * 100).toFixed(1)}%`;
  elements.alertDesc.textContent = `Within-camera ROI mass: Front ${pct(aotrFront)}, Wrist ${pct(aotrWrist)}. Interpret relative to ROI area and visibility. Heatmaps alone cannot establish camera use or a failure cause; the action trace re-queries every frame.`;
}

// Sidebar Trajectory Readout
function updateJointTable(fIdx) {
  const trajs = state.data.trajectories;
  const gt = trajs.gt_actions[fIdx] || [];
  const pred = trajs.pred_actions[fIdx] || [];
  const names = state.data.metadata.joint_names;

  let html = "";
  names.forEach((name, i) => {
    const g = gt[i] !== undefined ? gt[i].toFixed(2) : "--";
    const p = pred[i] !== undefined ? pred[i].toFixed(2) : "--";
    const diff = (gt[i] !== undefined && pred[i] !== undefined) ? (p - g).toFixed(2) : "--";
    const isBigDiff = Math.abs(p - g) > 5.0;
    html += `<tr>
      <td><strong>${name}</strong></td>
      <td style="color:#38bdf8">${g}</td>
      <td style="color:#f85149">${p}</td>
      <td style="color:${isBigDiff ? '#f85149' : '#8b949e'}">${diff}</td>
    </tr>`;
  });
  elements.tbodyJoints.innerHTML = html;
}

function updateLatentCards(frameInfo) {
  const lStats = frameInfo.layer_latents;
  if (!lStats) return;

  for (let i = 0; i < 4; i++) {
    const key = `layer_${i}`;
    const info = lStats[key];
    if (!info) continue;

    const normEl = document.getElementById(`val-l${i}-norm`);
    const cosEl = document.getElementById(`val-l${i}-cossim`);
    const barEl = document.getElementById(`bar-l${i}`);

    if (normEl) normEl.textContent = info.mean_norm.toFixed(2);
    if (cosEl && i > 0) cosEl.textContent = info.cosine_sim_with_prev.toFixed(2);
    if (barEl) {
      const pct = Math.min(100, Math.max(10, (info.mean_norm / 15.0) * 100));
      barEl.style.width = `${pct}%`;
    }
  }

  // Entropies
  if (frameInfo.entropy_per_layer) {
    frameInfo.entropy_per_layer.forEach((ent, i) => {
      const badge = document.getElementById(`badge-ent-l${i}`);
      if (badge) badge.textContent = `L${i} Entropy: ${ent.toFixed(2)}`;
    });
  }
}

// Canvas Charts
function drawTrajectoryChart() {
  const canvas = elements.canvasTrajectories;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.data) return;

  const trajs = state.data.trajectories;
  const gt = trajs.gt_actions;
  const pred = trajs.pred_actions;
  const total = gt.length;
  if (total === 0) return;

  // Plot Joint 5 (Gripper) or Joint 1 (Lift)
  const jointIdx = 5; // Gripper
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < total; i++) {
    minVal = Math.min(minVal, gt[i][jointIdx], pred[i][jointIdx]);
    maxVal = Math.max(maxVal, gt[i][jointIdx], pred[i][jointIdx]);
  }
  const range = (maxVal - minVal) || 1.0;

  // Background grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = 20; y < h; y += 40) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();

  // Plot GT (Cyan)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < total; i++) {
    const x = (i / (total - 1)) * w;
    const y = h - 20 - ((gt[i][jointIdx] - minVal) / range) * (h - 40);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Plot Pred (Coral)
  ctx.strokeStyle = "#f85149";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (let i = 0; i < total; i++) {
    const x = (i / (total - 1)) * w;
    const y = h - 20 - ((pred[i][jointIdx] - minVal) / range) * (h - 40);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Time Cursor Indicator
  const cursorX = (state.currentFrame / (total - 1)) * w;
  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();

  // Title in canvas
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("Gripper Open/Close Trajectory (Joint 5)", 10, 16);
}

function drawAoTRTimelineChart() {
  const canvas = elements.canvasAoTRTimeline;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.data) return;

  const total = state.data.metadata.total_frames;
  const aotrFront = state.aotrTimeline.front;
  const aotrWrist = state.aotrTimeline.wrist;

  // Background grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  [0.25, 0.5, 0.75, 1.0].forEach(val => {
    const y = h - 20 - val * (h - 30);
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  });
  ctx.stroke();

  // Plot Front-Left AoTR (Blue)
  if (aotrFront && aotrFront.length > 0) {
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < total; i++) {
      if (aotrFront[i] !== null) {
        const x = (i / (total - 1)) * w;
        const y = h - 20 - Math.min(1.0, aotrFront[i]) * (h - 30);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  // Plot Wrist AoTR (Purple)
  if (aotrWrist && aotrWrist.length > 0) {
    ctx.strokeStyle = "#a371f7";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < total; i++) {
      if (aotrWrist[i] !== null) {
        const x = (i / (total - 1)) * w;
        const y = h - 20 - Math.min(1.0, aotrWrist[i]) * (h - 30);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  // Current Frame Marker
  const cursorX = (state.currentFrame / (total - 1)) * w;
  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, h);
  ctx.stroke();
}

function drawChunkRolloutChart() {
  const canvas = elements.canvasChunkRollout;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!state.data) return;

  const chunks = state.data.trajectories.pred_chunks || [];
  if (chunks.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("No subsampled chunk rollouts found in data file.", 20, 40);
    return;
  }

  // Find nearest chunk to current frame
  let bestChunk = chunks[0];
  let minDiff = Infinity;
  chunks.forEach(item => {
    const diff = Math.abs(item.frame_idx - state.currentFrame);
    if (diff < minDiff) {
      minDiff = diff;
      bestChunk = item;
    }
  });

  const cData = bestChunk.chunk; // [50, 6]
  const cLen = cData.length;

  ctx.fillStyle = "#f0f6fc";
  ctx.font = "12px JetBrains Mono, monospace";
  const stride = state.data.metadata.chunk_sample_stride || 2;
  ctx.fillText(`Predicted at Frame ${bestChunk.frame_idx}: query offsets 0..${(cLen - 1) * stride}, stride ${stride}`, 20, 24);

  // Plot each joint in the chunk with distinct colors
  const jointColors = ["#38bdf8", "#3fb950", "#eab308", "#a371f7", "#ec4899", "#f85149"];
  for (let j = 0; j < 6; j++) {
    ctx.strokeStyle = jointColors[j];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let step = 0; step < cLen; step++) {
      const x = 30 + (step / (cLen - 1)) * (w - 60);
      const val = cData[step][j];
      const y = h / 2 - val * 2.0; // scaled
      if (step === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

// Start
window.addEventListener("DOMContentLoaded", init);
