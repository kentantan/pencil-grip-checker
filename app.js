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
const K_NEIGHBORS = 5;
const BAD_CONFIRMATION_MS = 3000;
const ALERT_COOLDOWN_MS = 10000;
const ALERT_VISIBLE_MS = 1800;
const HISTORY_LENGTH = 15;

// badThresholdが高いほど、BADにかなり近い場合だけ警告する。
const SENSITIVITY = {
  1: { label: "かなり緩い", badThreshold: 0.78 },
  2: { label: "緩い", badThreshold: 0.68 },
  3: { label: "標準", badThreshold: 0.58 },
  4: { label: "厳しい", badThreshold: 0.52 },
  5: { label: "かなり厳しい", badThreshold: 0.47 },
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
  registerGoodButton: document.querySelector("#registerGoodButton"),
  registerBadButton: document.querySelector("#registerBadButton"),
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

let registeringType = null; // "good" | "bad" | null
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

function sampleCount(type) {
  const key = type === "good" ? "goodSamples" : "badSamples";
  return Array.isArray(profile?.[key]) ? profile[key].length : 0;
}

function hasAnyProfile() {
  return sampleCount("good") > 0 || sampleCount("bad") > 0;
}

function hasCompleteProfile() {
  return sampleCount("good") >= MIN_REGISTRATION_SAMPLES
    && sampleCount("bad") >= MIN_REGISTRATION_SAMPLES;
}

function updateButtons() {
  const cameraActive = Boolean(mediaStream);
  const registering = Boolean(registeringType);

  elements.startCameraButton.disabled = cameraActive;
  elements.stopCameraButton.disabled = !cameraActive;
  elements.switchCameraButton.disabled = !cameraActive || registering;
  elements.registerGoodButton.disabled = !cameraActive || registering;
  elements.registerBadButton.disabled = !cameraActive || registering;
  elements.monitorButton.disabled = !cameraActive || !hasCompleteProfile() || registering;
  elements.clearProfileButton.disabled = !hasAnyProfile() || registering;
  elements.monitorButton.textContent = monitoring ? "見守りを停止" : "見守りを開始";
}

function formatRegisteredAt(value) {
  if (!value) return "未登録";
  return new Date(value).toLocaleString("ja-JP");
}

function updateProfileInfo() {
  const goodCount = sampleCount("good");
  const badCount = sampleCount("bad");

  if (!hasAnyProfile()) {
    elements.profileInfo.textContent = "GOOD・BADとも未登録";
    return;
  }

  const readyText = hasCompleteProfile() ? "見守り可能" : "両方を登録してください";
  elements.profileInfo.textContent =
    `GOOD ${goodCount}件（${formatRegisteredAt(profile.goodRegisteredAt)}）／`
    + `BAD ${badCount}件（${formatRegisteredAt(profile.badRegisteredAt)}）／${readyText}`;
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
  const source = worldLandmarks?.length === 21 ? worldLandmarks : normalizedLandmarks;
  const points = canonicalizeLandmarks(source);
  if (!points) return null;

  const features = [];

  for (const point of points) {
    features.push(point.x, point.y, point.z);
  }

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

function calculateClassScale(samples) {
  if (!Array.isArray(samples) || samples.length < 2) return 0.03;

  const nearestDistances = samples.map((sample, sampleIndex) => {
    let nearest = Number.POSITIVE_INFINITY;
    for (let compareIndex = 0; compareIndex < samples.length; compareIndex += 1) {
      if (sampleIndex === compareIndex) continue;
      nearest = Math.min(nearest, featureDistance(sample, samples[compareIndex]));
    }
    return nearest;
  }).filter(Number.isFinite);

  // 連続フレームは似やすいため、極端に小さい尺度にならないよう下限を設ける。
  return Math.max(0.018, percentile(nearestDistances, 0.90) * 1.8 + 0.004);
}

function kNearestAverageDistance(feature, samples, k = K_NEIGHBORS) {
  if (!Array.isArray(samples) || samples.length === 0) {
    return Number.POSITIVE_INFINITY;
  }

  const distances = samples
    .map((sample) => featureDistance(feature, sample))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  const selected = distances.slice(0, Math.min(k, distances.length));
  if (!selected.length) return Number.POSITIVE_INFINITY;
  return selected.reduce((sum, value) => sum + value, 0) / selected.length;
}

function normalizeLoadedProfile(loaded) {
  if (!loaded) return null;

  if (loaded.schemaVersion === 2) {
    const goodSamples = Array.isArray(loaded.goodSamples) ? loaded.goodSamples : [];
    const badSamples = Array.isArray(loaded.badSamples) ? loaded.badSamples : [];
    return {
      ...loaded,
      goodSamples,
      badSamples,
      goodScale: Number.isFinite(loaded.goodScale)
        ? loaded.goodScale
        : calculateClassScale(goodSamples),
      badScale: Number.isFinite(loaded.badScale)
        ? loaded.badScale
        : calculateClassScale(badSamples),
    };
  }

  // 旧版の「正しい持ち方のみ」の登録はGOODとして引き継ぐ。
  if (Array.isArray(loaded.samples) && loaded.samples.length > 0) {
    return {
      id: loaded.id,
      schemaVersion: 2,
      createdAt: loaded.createdAt ?? new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      goodRegisteredAt: loaded.createdAt ?? new Date().toISOString(),
      badRegisteredAt: null,
      goodSamples: loaded.samples,
      badSamples: [],
      goodScale: calculateClassScale(loaded.samples),
      badScale: 0.03,
    };
  }

  return null;
}

function classifyFeature(feature) {
  if (!hasCompleteProfile()) return null;

  const goodRaw = kNearestAverageDistance(feature, profile.goodSamples);
  const badRaw = kNearestAverageDistance(feature, profile.badSamples);
  const goodDistance = goodRaw / Math.max(profile.goodScale, 1e-6);
  const badDistance = badRaw / Math.max(profile.badScale, 1e-6);
  const denominator = goodDistance + badDistance;

  // GOODに近いほど0、BADに近いほど1。
  const badProbability = denominator > 1e-8 ? goodDistance / denominator : 0.5;
  return {
    goodRaw,
    badRaw,
    goodDistance,
    badDistance,
    badProbability,
  };
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
  const classification = classifyFeature(feature);
  if (!classification) return;

  const setting = SENSITIVITY[elements.sensitivity.value];
  const isBad = classification.badProbability >= setting.badThreshold;
  const badPercent = Math.round(classification.badProbability * 100);

  elements.scoreText.textContent = `BAD ${badPercent}%`;
  decisionHistory.push(isBad);
  if (decisionHistory.length > HISTORY_LENGTH) decisionHistory.shift();

  const badCount = decisionHistory.filter(Boolean).length;
  const badRatio = badCount / decisionHistory.length;
  const stableBad = decisionHistory.length >= 8 && badRatio >= 0.65;

  if (stableBad) {
    if (badStateStartedAt === null) badStateStartedAt = performance.now();
    const elapsed = performance.now() - badStateStartedAt;
    const remaining = Math.max(0, Math.ceil((BAD_CONFIRMATION_MS - elapsed) / 1000));

    if (elapsed >= BAD_CONFIRMATION_MS) {
      setStatus("BADに近い持ち方", "bad");
      showWarning();
    } else {
      setStatus(`確認中 ${remaining}秒`, "register");
    }
  } else {
    badStateStartedAt = null;
    setStatus(isBad ? "動きを確認中" : "GOODに近い持ち方", isBad ? "idle" : "good");
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

function registrationLabel(type) {
  return type === "good" ? "GOOD" : "BAD";
}

function processRegistration(feature, now) {
  if (!registeringType || registrationStartedAt <= 0
      || now - lastRegistrationSampleAt < REGISTRATION_INTERVAL_MS) return;

  lastRegistrationSampleAt = now;
  registrationSamples.push(feature);

  const elapsedSeconds = (now - registrationStartedAt) / 1000;
  const remaining = Math.max(0, Math.ceil(REGISTRATION_SECONDS - elapsedSeconds));
  setStatus(`${registrationLabel(registeringType)}登録中 残り${remaining}秒`, "register");
  elements.scoreText.textContent = `${registrationSamples.length}件`;

  if (elapsedSeconds >= REGISTRATION_SECONDS) {
    finishRegistration().catch(handleError);
  }
}

async function finishRegistration() {
  if (!registeringType) return;
  const completedType = registeringType;
  registeringType = null;
  registrationStartedAt = 0;
  clearTimeout(registrationEndTimer);
  registrationEndTimer = null;

  if (registrationSamples.length < MIN_REGISTRATION_SAMPLES) {
    setStatus(`${registrationLabel(completedType)}登録失敗`, "bad");
    alert("有効な手の検出が少なすぎました。手全体が映る位置で、もう一度登録してください。");
    updateButtons();
    return;
  }

  const nowIso = new Date().toISOString();
  const baseProfile = profile ?? {
    schemaVersion: 2,
    createdAt: nowIso,
    goodSamples: [],
    badSamples: [],
    goodScale: 0.03,
    badScale: 0.03,
    goodRegisteredAt: null,
    badRegisteredAt: null,
  };

  if (completedType === "good") {
    baseProfile.goodSamples = registrationSamples;
    baseProfile.goodScale = calculateClassScale(registrationSamples);
    baseProfile.goodRegisteredAt = nowIso;
  } else {
    baseProfile.badSamples = registrationSamples;
    baseProfile.badScale = calculateClassScale(registrationSamples);
    baseProfile.badRegisteredAt = nowIso;
  }

  baseProfile.schemaVersion = 2;
  baseProfile.updatedAt = nowIso;
  profile = baseProfile;
  await saveProfile(profile);

  setStatus(`${registrationLabel(completedType)}登録完了`, "good");
  elements.scoreText.textContent = `${registrationSamples.length}件`;
  updateProfileInfo();
  updateButtons();
}

async function startRegistration(type) {
  if (!mediaStream || registeringType) return;
  if (type !== "good" && type !== "bad") return;

  registeringType = type;
  registrationStartedAt = 0;
  monitoring = false;
  resetDecisionState();
  updateButtons();

  const readyText = type === "good"
    ? "正しく持って準備"
    : "普段戻りやすい持ち方で準備";

  for (let count = 3; count >= 1; count -= 1) {
    elements.countdown.hidden = false;
    elements.countdown.textContent = String(count);
    setStatus(readyText, "register");
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
    setStatus(hasCompleteProfile() ? "見守り待機" : "GOOD・BADを登録", "idle");
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
  registeringType = null;
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
  monitoring = wasMonitoring && hasCompleteProfile();
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
      if (registeringType) {
        setStatus("手を枠内へ", "bad");
      } else if (monitoring) {
        setStatus("手を探しています", "idle");
      }
      badStateStartedAt = null;
      return;
    }

    const feature = extractFeatureVector(result.worldLandmarks?.[0], result.landmarks[0]);
    if (!feature) return;

    if (registeringType) {
      processRegistration(feature, now);
    } else if (monitoring && hasCompleteProfile()) {
      evaluateFeature(feature);
    } else if (hasCompleteProfile()) {
      setStatus("見守り待機", "idle");
    } else {
      setStatus("GOOD・BADを登録", "idle");
    }
  } catch (error) {
    console.error("フレーム処理エラー", error);
    setStatus("処理エラー", "bad");
  }
}

function toggleMonitoring() {
  if (!hasCompleteProfile() || !mediaStream) return;
  monitoring = !monitoring;
  resetDecisionState();
  setStatus(monitoring ? "見守り中" : "見守り待機", monitoring ? "good" : "idle");
  updateButtons();
}

async function clearProfile() {
  const confirmed = window.confirm("端末内に保存したGOOD・BADの手座標データを削除しますか？");
  if (!confirmed) return;

  monitoring = false;
  await deleteProfile();
  profile = null;
  resetDecisionState();
  updateProfileInfo();
  setStatus(mediaStream ? "GOOD・BADを登録" : "準備前", "idle");
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

  profile = normalizeLoadedProfile(await loadProfile());
  if (profile) await saveProfile(profile);
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
elements.registerGoodButton.addEventListener("click", () => startRegistration("good").catch(handleError));
elements.registerBadButton.addEventListener("click", () => startRegistration("bad").catch(handleError));
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
