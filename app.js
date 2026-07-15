import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
} from "./vendor/mediapipe/vision_bundle.mjs";
import { deleteProfile, loadProfile, saveProfile } from "./db.js";

const MODEL_PATH = "./models/hand_landmarker.task";
const WASM_PATH = "./vendor/mediapipe/wasm";

const DETECTION_INTERVAL_MS = 100; // 約10fps。スマホの発熱を抑える
const REGISTRATION_SECONDS = 10;
const REGISTRATION_INTERVAL_MS = 200; // 10秒で最大50件
const MIN_REGISTRATION_SAMPLES = 20;
const BAD_CONFIRMATION_MS = 3000;
const ALERT_COOLDOWN_MS = 10000;
const ALERT_VISIBLE_MS = 1800;
const HISTORY_LENGTH = 15;

const SENSITIVITY = {
  1: { label: "かなり緩い", multiplier: 1.65 },
  2: { label: "緩い", multiplier: 1.35 },
  3: { label: "標準", multiplier: 1.12 },
  4: { label: "厳しい", multiplier: 0.95 },
  5: { label: "かなり厳しい", multiplier: 0.80 },
};

const elements = {
  video: document.querySelector("#camera"),
  canvas: document.querySelector("#overlay"),
  cameraStage: document.querySelector("#cameraStage"),
  warningOverlay: document.querySelector("#warningOverlay"),
  countdown: document.querySelector("#countdown"),
  statusText: document.querySelector("#statusText"),
  scoreText: document.querySelector("#scoreText"),
  runtimeInfo: document.querySelector("#runtimeInfo"),
  profileInfo: document.querySelector("#profileInfo"),
  startCameraButton: document.querySelector("#startCameraButton"),
  stopCameraButton: document.querySelector("#stopCameraButton"),
  switchCameraButton: document.querySelector("#switchCameraButton"),
  registerButton: document.querySelector("#registerButton"),
  monitorButton: document.querySelector("#monitorButton"),
  clearProfileButton: document.querySelector("#clearProfileButton"),
  sensitivity: document.querySelector("#sensitivity"),
  sensitivityText: document.querySelector("#sensitivityText"),
};

const canvasContext = elements.canvas.getContext("2d");
const drawingUtils = new DrawingUtils(canvasContext);

let handLandmarker = null;
let mediaStream = null;
let animationFrameId = null;
let profile = null;
let cameraFacingMode = "environment";
let lastDetectionAt = 0;
let lastVideoTime = -1;
let monitoring = false;
let wakeLock = null;
let audioContext = null;

let registering = false;
let registrationStartedAt = 0;
let lastRegistrationSampleAt = 0;
let registrationSamples = [];
let registrationEndTimer = null;

let decisionHistory = [];
let badStateStartedAt = null;
let lastAlertAt = 0;
let warningTimer = null;

function setStatus(text, kind = "idle") {
  elements.statusText.textContent = text;
  elements.statusText.className = `status-value status-${kind}`;
}

function setBusy(button, busy, busyText) {
  if (!button.dataset.originalText) {
    button.dataset.originalText = button.textContent.trim();
  }
  button.disabled = busy;
  button.textContent = busy ? busyText : button.dataset.originalText;
}

function updateButtons() {
  const cameraActive = Boolean(mediaStream);
  const hasProfile = Boolean(profile);

  elements.startCameraButton.disabled = cameraActive;
  elements.stopCameraButton.disabled = !cameraActive;
  elements.switchCameraButton.disabled = !cameraActive || registering;
  elements.registerButton.disabled = !cameraActive || registering;
  elements.monitorButton.disabled = !cameraActive || !hasProfile || registering;
  elements.clearProfileButton.disabled = !hasProfile || registering;
  elements.monitorButton.textContent = monitoring ? "見守りを停止" : "見守りを開始";
}

function updateProfileInfo() {
  if (!profile) {
    elements.profileInfo.textContent = "登録データなし";
    return;
  }

  const created = new Date(profile.createdAt).toLocaleString("ja-JP");
  elements.profileInfo.textContent =
    `${profile.samples.length}件登録済み（${created}）／基準値 ${profile.baseThreshold.toFixed(4)}`;
}

function updateSensitivityLabel() {
  const setting = SENSITIVITY[elements.sensitivity.value];
  elements.sensitivityText.textContent = setting.label;
}

function vecSubtract(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function vecDot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function vecCross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function vecLength(v) {
  return Math.hypot(v.x, v.y, v.z);
}

function vecScale(v, scale) {
  return { x: v.x * scale, y: v.y * scale, z: v.z * scale };
}

function vecNormalize(v) {
  const length = vecLength(v);
  if (!Number.isFinite(length) || length < 1e-8) return null;
  return vecScale(v, 1 / length);
}

function distance3d(a, b) {
  return vecLength(vecSubtract(a, b));
}

function angleAt(a, b, c) {
  const ba = vecSubtract(a, b);
  const bc = vecSubtract(c, b);
  const denominator = vecLength(ba) * vecLength(bc);
  if (denominator < 1e-8) return 0;
  const cosine = Math.max(-1, Math.min(1, vecDot(ba, bc) / denominator));
  return Math.acos(cosine) / Math.PI;
}

/**
 * 手首を原点、掌の横方向・縦方向を基準軸にして3D座標を正規化する。
 * 平行移動・拡大縮小・カメラに対する手の回転の影響を減らす。
 */
function canonicalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  const origin = landmarks[0];
  const acrossPalm = vecSubtract(landmarks[5], landmarks[17]);
  const towardFingers = vecSubtract(landmarks[9], origin);

  const xAxis = vecNormalize(acrossPalm);
  if (!xAxis) return null;

  const projection = vecScale(xAxis, vecDot(towardFingers, xAxis));
  const yAxis = vecNormalize(vecSubtract(towardFingers, projection));
  if (!yAxis) return null;

  const zAxis = vecNormalize(vecCross(xAxis, yAxis));
  if (!zAxis) return null;

  const palmLength = distance3d(origin, landmarks[9]);
  const palmWidth = distance3d(landmarks[5], landmarks[17]);
  const scale = (palmLength + palmWidth) / 2;
  if (!Number.isFinite(scale) || scale < 1e-8) return null;

  return landmarks.map((point) => {
    const relative = vecSubtract(point, origin);
    return {
      x: vecDot(relative, xAxis) / scale,
      y: vecDot(relative, yAxis) / scale,
      z: vecDot(relative, zAxis) / scale,
    };
  });
}

function extractFeatureVector(worldLandmarks, normalizedLandmarks) {
  // worldLandmarksを優先し、端末によって取れない場合は画面座標へフォールバック
  const source = worldLandmarks?.length === 21 ? worldLandmarks : normalizedLandmarks;
  const points = canonicalizeLandmarks(source);
  if (!points) return null;

  const features = [];

  // 21点の正規化3D座標
  for (const point of points) {
    features.push(point.x, point.y, point.z);
  }

  // 各指の曲がり角度。0～1へ正規化
  const angleTriples = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4],
    [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12],
    [0, 13, 14], [13, 14, 15], [14, 15, 16],
    [0, 17, 18], [17, 18, 19], [18, 19, 20],
  ];
  for (const [a, b, c] of angleTriples) {
    features.push(angleAt(points[a], points[b], points[c]) * 0.7);
  }

  // 鉛筆把持に関係しやすい指同士の距離
  const distancePairs = [
    [4, 8], [4, 6], [4, 5], [4, 10], [4, 12],
    [8, 10], [8, 12], [8, 16], [8, 20],
    [5, 9], [9, 13], [13, 17],
  ];
  for (const [a, b] of distancePairs) {
    features.push(distance3d(points[a], points[b]) * 0.8);
  }

  return features.every(Number.isFinite) ? features : null;
}

function featureDistance(a, b) {
  if (!a || !b || a.length !== b.length) return Number.POSITIVE_INFINITY;
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    const difference = a[index] - b[index];
    sum += difference * difference;
  }
  return Math.sqrt(sum / a.length);
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(ratio * sorted.length) - 1));
  return sorted[index];
}

function calculateBaseThreshold(samples) {
  const nearestDistances = samples.map((sample, sampleIndex) => {
    let nearest = Number.POSITIVE_INFINITY;
    for (let compareIndex = 0; compareIndex < samples.length; compareIndex += 1) {
      if (sampleIndex === compareIndex) continue;
      nearest = Math.min(nearest, featureDistance(sample, samples[compareIndex]));
    }
    return nearest;
  }).filter(Number.isFinite);

  // 登録中の自然な揺れの95%点を基にし、未知の筆記動作分の余裕を持たせる。
  const observedVariation = percentile(nearestDistances, 0.95);
  return Math.max(0.028, observedVariation * 2.2 + 0.008);
}

function currentThreshold() {
  if (!profile) return Number.POSITIVE_INFINITY;
  const setting = SENSITIVITY[elements.sensitivity.value];
  return profile.baseThreshold * setting.multiplier;
}

function nearestProfileDistance(feature) {
  if (!profile?.samples?.length) return Number.POSITIVE_INFINITY;
  let nearest = Number.POSITIVE_INFINITY;
  for (const sample of profile.samples) {
    nearest = Math.min(nearest, featureDistance(feature, sample));
  }
  return nearest;
}

async function createHandLandmarker() {
  elements.runtimeInfo.textContent = "AIモデルを読み込み中…";
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  const commonOptions = {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, commonOptions);
    elements.runtimeInfo.textContent = "MediaPipe Hand Landmarker／GPU優先／端末内処理";
  } catch (gpuError) {
    console.warn("GPU初期化に失敗したためCPUへ切り替えます。", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...commonOptions,
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    });
    elements.runtimeInfo.textContent = "MediaPipe Hand Landmarker／CPU／端末内処理";
  }
}

function initializeAudio() {
  if (audioContext) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (AudioContextClass) {
    audioContext = new AudioContextClass();
  }
}

function playAlertSound() {
  if (!audioContext) return;
  if (audioContext.state === "suspended") {
    audioContext.resume().catch(() => {});
  }

  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = 660;
  gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.12, audioContext.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.22);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.24);
}

function showWarning() {
  const now = performance.now();
  if (now - lastAlertAt < ALERT_COOLDOWN_MS) return;
  lastAlertAt = now;

  elements.warningOverlay.hidden = false;
  navigator.vibrate?.([120, 80, 120]);
  playAlertSound();

  clearTimeout(warningTimer);
  warningTimer = setTimeout(() => {
    elements.warningOverlay.hidden = true;
  }, ALERT_VISIBLE_MS);
}

function resetDecisionState() {
  decisionHistory = [];
  badStateStartedAt = null;
  elements.scoreText.textContent = "--";
  elements.warningOverlay.hidden = true;
}

function evaluateFeature(feature) {
  const distance = nearestProfileDistance(feature);
  const threshold = currentThreshold();
  const isGood = distance <= threshold;

  elements.scoreText.textContent = `${distance.toFixed(4)} / ${threshold.toFixed(4)}`;
  decisionHistory.push(isGood);
  if (decisionHistory.length > HISTORY_LENGTH) decisionHistory.shift();

  const goodCount = decisionHistory.filter(Boolean).length;
  const goodRatio = goodCount / decisionHistory.length;
  const stableBad = decisionHistory.length >= 8 && goodRatio < 0.35;

  if (stableBad) {
    if (badStateStartedAt === null) badStateStartedAt = performance.now();
    const elapsed = performance.now() - badStateStartedAt;
    const remaining = Math.max(0, Math.ceil((BAD_CONFIRMATION_MS - elapsed) / 1000));

    if (elapsed >= BAD_CONFIRMATION_MS) {
      setStatus("持ち方を確認", "bad");
      showWarning();
    } else {
      setStatus(`確認中 ${remaining}秒`, "register");
    }
  } else {
    badStateStartedAt = null;
    setStatus(isGood ? "いい持ち方" : "動きを確認中", isGood ? "good" : "idle");
  }
}

function drawResults(result) {
  canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  if (!result.landmarks?.length) return;

  const landmarks = result.landmarks[0];
  drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
    color: "#62d58a",
    lineWidth: 4,
  });
  drawingUtils.drawLandmarks(landmarks, {
    color: "#ffffff",
    fillColor: "#c8783e",
    lineWidth: 2,
    radius: 4,
  });
}

function processRegistration(feature, now) {
  if (!registering || registrationStartedAt <= 0 || now - lastRegistrationSampleAt < REGISTRATION_INTERVAL_MS) return;
  lastRegistrationSampleAt = now;
  registrationSamples.push(feature);

  const elapsedSeconds = (now - registrationStartedAt) / 1000;
  const remaining = Math.max(0, Math.ceil(REGISTRATION_SECONDS - elapsedSeconds));
  setStatus(`登録中 残り${remaining}秒`, "register");
  elements.scoreText.textContent = `${registrationSamples.length}件`;

  if (elapsedSeconds >= REGISTRATION_SECONDS) {
    finishRegistration().catch(handleError);
  }
}

async function finishRegistration() {
  if (!registering) return;
  registering = false;
  registrationStartedAt = 0;
  clearTimeout(registrationEndTimer);
  registrationEndTimer = null;

  if (registrationSamples.length < MIN_REGISTRATION_SAMPLES) {
    setStatus("登録失敗：手が見えません", "bad");
    alert("有効な手の検出が少なすぎました。手全体が映る位置で、もう一度登録してください。");
    updateButtons();
    return;
  }

  const baseThreshold = calculateBaseThreshold(registrationSamples);
  profile = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    baseThreshold,
    samples: registrationSamples,
  };
  await saveProfile(profile);

  setStatus("登録完了", "good");
  elements.scoreText.textContent = `${registrationSamples.length}件`;
  updateProfileInfo();
  updateButtons();
}

async function startRegistration() {
  if (!mediaStream || registering) return;
  registering = true;
  registrationStartedAt = 0;
  monitoring = false;
  resetDecisionState();
  updateButtons();

  for (let count = 3; count >= 1; count -= 1) {
    elements.countdown.hidden = false;
    elements.countdown.textContent = String(count);
    setStatus("正しく持って準備", "register");
    await new Promise((resolve) => setTimeout(resolve, 850));
  }

  elements.countdown.textContent = "開始";
  await new Promise((resolve) => setTimeout(resolve, 450));
  elements.countdown.hidden = true;

  registrationSamples = [];
  registrationStartedAt = performance.now();
  lastRegistrationSampleAt = 0;
  registrationEndTimer = setTimeout(() => {
    finishRegistration().catch(handleError);
  }, REGISTRATION_SECONDS * 1000 + 300);
  updateButtons();
}

async function requestWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => {
      wakeLock = null;
    });
  } catch (error) {
    console.info("画面のスリープ防止を利用できません。", error);
  }
}

async function releaseWakeLock() {
  if (!wakeLock) return;
  try {
    await wakeLock.release();
  } catch {
    // 既に解放済みの場合は何もしない
  }
  wakeLock = null;
}

async function startCamera() {
  if (mediaStream) return;
  initializeAudio();
  setBusy(elements.startCameraButton, true, "カメラ準備中…");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: cameraFacingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 24, max: 30 },
      },
    });

    elements.video.srcObject = mediaStream;
    await elements.video.play();

    const width = elements.video.videoWidth || 1280;
    const height = elements.video.videoHeight || 720;
    elements.canvas.width = width;
    elements.canvas.height = height;
    elements.cameraStage.style.aspectRatio = `${width} / ${height}`;

    await requestWakeLock();
    setStatus(profile ? "見守り待機" : "登録してください", "idle");
    animationFrameId = requestAnimationFrame(renderLoop);
  } catch (error) {
    mediaStream = null;
    if (error?.name === "NotAllowedError") {
      throw new Error("カメラが許可されていません。ブラウザのサイト設定でカメラを許可してください。");
    }
    throw error;
  } finally {
    setBusy(elements.startCameraButton, false, "");
    updateButtons();
  }
}

async function stopCamera() {
  monitoring = false;
  registering = false;
  registrationStartedAt = 0;
  clearTimeout(registrationEndTimer);
  registrationEndTimer = null;
  resetDecisionState();

  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  elements.video.srcObject = null;
  canvasContext.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  await releaseWakeLock();
  setStatus("停止中", "idle");
  updateButtons();
}

async function switchCamera() {
  const wasMonitoring = monitoring;
  await stopCamera();
  cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
  await startCamera();
  monitoring = wasMonitoring && Boolean(profile);
  updateButtons();
}

function renderLoop(now) {
  animationFrameId = requestAnimationFrame(renderLoop);
  if (!handLandmarker || !mediaStream || elements.video.readyState < 2) return;
  if (now - lastDetectionAt < DETECTION_INTERVAL_MS) return;
  if (elements.video.currentTime === lastVideoTime) return;

  lastDetectionAt = now;
  lastVideoTime = elements.video.currentTime;

  try {
    const result = handLandmarker.detectForVideo(elements.video, now);
    drawResults(result);

    if (!result.landmarks?.length) {
      if (registering) {
        setStatus("手を枠内へ", "bad");
      } else if (monitoring) {
        setStatus("手を探しています", "idle");
      }
      badStateStartedAt = null;
      return;
    }

    const feature = extractFeatureVector(result.worldLandmarks?.[0], result.landmarks[0]);
    if (!feature) return;

    if (registering) {
      processRegistration(feature, now);
    } else if (monitoring && profile) {
      evaluateFeature(feature);
    } else if (profile) {
      setStatus("見守り待機", "idle");
    } else {
      setStatus("正しい持ち方を登録", "idle");
    }
  } catch (error) {
    console.error("フレーム処理エラー", error);
    setStatus("処理エラー", "bad");
  }
}

function toggleMonitoring() {
  if (!profile || !mediaStream) return;
  monitoring = !monitoring;
  resetDecisionState();
  setStatus(monitoring ? "見守り中" : "見守り待機", monitoring ? "good" : "idle");
  updateButtons();
}

async function clearProfile() {
  const confirmed = window.confirm("端末内に保存した手の座標データを削除しますか？");
  if (!confirmed) return;

  monitoring = false;
  await deleteProfile();
  profile = null;
  resetDecisionState();
  updateProfileInfo();
  setStatus(mediaStream ? "登録してください" : "準備前", "idle");
  updateButtons();
}

function handleError(error) {
  console.error(error);
  const message = error instanceof Error ? error.message : String(error);
  setStatus("エラー", "bad");
  alert(message);
}

async function initialize() {
  updateSensitivityLabel();

  if (!window.isSecureContext) {
    throw new Error("カメラ利用にはHTTPSが必要です。GitHub PagesのHTTPS URLから開いてください。");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("このブラウザはカメラAPIに対応していません。SafariまたはChromeの最新版を使用してください。");
  }
  if (!("indexedDB" in window)) {
    throw new Error("このブラウザは端末内保存（IndexedDB）に対応していません。");
  }

  profile = await loadProfile();
  updateProfileInfo();
  updateButtons();
  await createHandLandmarker();

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch((error) => {
      console.info("オフラインキャッシュを登録できませんでした。", error);
    });
  }
}

elements.startCameraButton.addEventListener("click", () => startCamera().catch(handleError));
elements.stopCameraButton.addEventListener("click", () => stopCamera().catch(handleError));
elements.switchCameraButton.addEventListener("click", () => switchCamera().catch(handleError));
elements.registerButton.addEventListener("click", () => startRegistration().catch(handleError));
elements.monitorButton.addEventListener("click", toggleMonitoring);
elements.clearProfileButton.addEventListener("click", () => clearProfile().catch(handleError));
elements.sensitivity.addEventListener("input", updateSensitivityLabel);

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && mediaStream) {
    requestWakeLock();
  }
});

window.addEventListener("pagehide", () => {
  mediaStream?.getTracks().forEach((track) => track.stop());
});

initialize().catch(handleError);
