/**
 * HealthSense — Professional Healthcare Management Platform Controller
 * Manages Authentication & Registration, Navigation, Real-Time SSE Telemetry,
 * AI ECG 1D-CNN Inference, Medical Reports Generation, and Interactive History.
 */

// Application Global State
const appState = {
  isLoggedIn: false,
  currentUser: null,
  currentPage: 'dashboard-page',
  patientName: 'Prajkta Patil',
  patientId: 'PNT-2026-0001',
  apiConnected: false,
  
  // ECG Beat Analysis
  currentSamples: [],
  currentLabel: 'Known Normal Beat (#0)',
  gain: 1.0,
  isAnalyzing: false,
  lastPrediction: {
    prediction: 'NORMAL',
    abnormal_probability: 0.008877,
    confidence: 0.991123,
    threshold: 0.30,
    class_id: 0,
    status: 'SUCCESS',
    timestamp: new Date().toLocaleTimeString()
  },

  // Live Telemetry Stream
  isStreaming: true,
  eventSource: null,
  liveBuffer: [],
  maxLivePoints: 500,
  currentVitals: {
    heartRate: 72,
    spo2: 98.0,
    temperature: 36.6,
    ecgStatus: 'NORMAL'
  }
};

// Built-in Verified Sample Cache
const BUILTIN_SAMPLES = {
  normal: null,
  abnormal: null
};

// DOM References
const DOM = {
  // Auth Screen & Cards
  authScreen: document.getElementById('auth-screen'),
  loginCard: document.getElementById('login-card'),
  registerCard: document.getElementById('register-card'),
  loginAlertBox: document.getElementById('login-alert-box'),
  loginAlertText: document.getElementById('login-alert-text'),
  registerAlertBox: document.getElementById('register-alert-box'),
  registerAlertText: document.getElementById('register-alert-text'),
  
  // Login Form
  loginForm: document.getElementById('login-form'),
  loginEmailInput: document.getElementById('login-email'),
  loginPasswordInput: document.getElementById('login-password'),
  loginSubmitBtn: document.getElementById('login-submit-btn'),
  passwordToggleBtn: document.getElementById('password-toggle-btn'),
  passwordEyeIcon: document.getElementById('password-eye-icon'),
  forgotPasswordLink: document.getElementById('forgot-password-link'),
  gotoRegisterBtn: document.getElementById('goto-register-btn'),
  
  // Register Form
  registerForm: document.getElementById('register-form'),
  regNameInput: document.getElementById('reg-name'),
  regEmailInput: document.getElementById('reg-email'),
  regPasswordInput: document.getElementById('reg-password'),
  regPasswordToggleBtn: document.getElementById('reg-password-toggle-btn'),
  regPasswordEyeIcon: document.getElementById('reg-password-eye-icon'),
  regConfirmPasswordInput: document.getElementById('reg-confirm-password'),
  regConfirmPasswordToggleBtn: document.getElementById('reg-confirm-password-toggle-btn'),
  regConfirmPasswordEyeIcon: document.getElementById('reg-confirm-password-eye-icon'),
  regPhoneInput: document.getElementById('reg-phone'),
  regAgeInput: document.getElementById('reg-age'),
  regGenderSelect: document.getElementById('reg-gender'),
  registerSubmitBtn: document.getElementById('register-submit-btn'),
  gotoLoginBtn: document.getElementById('goto-login-btn'),
  
  // Navigation & Shell
  appSidebar: document.getElementById('app-sidebar'),
  sidebarToggleBtn: document.getElementById('sidebar-toggle-btn'),
  navButtons: document.querySelectorAll('.nav-item-btn[data-page]'),
  pageViews: document.querySelectorAll('.page-view'),
  logoutBtn: document.getElementById('logout-btn'),
  dropdownLogoutBtn: document.getElementById('dropdown-logout-btn'),
  
  // Topbar & Profile
  globalSearchInput: document.getElementById('global-search-input'),
  topbarApiDot: document.getElementById('topbar-api-dot'),
  topbarApiText: document.getElementById('topbar-api-text'),
  notificationBtn: document.getElementById('notification-btn'),
  notificationDropdown: document.getElementById('notification-dropdown'),
  markReadBtn: document.getElementById('mark-read-btn'),
  userProfileToggle: document.getElementById('user-profile-toggle'),
  profileDropdownMenu: document.getElementById('profile-dropdown-menu'),
  topbarUserName: document.getElementById('topbar-user-name'),
  sidebarPatientName: document.getElementById('sidebar-patient-name'),
  dashboardGreetingName: document.getElementById('dashboard-greeting-name'),
  dropdownSettingsBtn: document.getElementById('dropdown-settings-btn'),
  dropdownReportsBtn: document.getElementById('dropdown-reports-btn'),

  // Dashboard Page
  dashSyncTime: document.getElementById('dashboard-sync-time'),
  dashQuickAnalyzeBtn: document.getElementById('dash-quick-analyze-btn'),
  dashHrVal: document.getElementById('dash-hr-val'),
  dashHrStatus: document.getElementById('dash-hr-status'),
  dashSpo2Val: document.getElementById('dash-spo2-val'),
  dashSpo2Status: document.getElementById('dash-spo2-status'),
  dashTempVal: document.getElementById('dash-temp-val'),
  dashTempStatus: document.getElementById('dash-temp-status'),
  dashOverallTitle: document.getElementById('dash-overall-title'),
  dashOverallMsg: document.getElementById('dash-overall-msg'),
  dashRptId: document.getElementById('dash-rpt-id'),
  dashRptDate: document.getElementById('dash-rpt-date'),
  dashRptTime: document.getElementById('dash-rpt-time'),
  dashRptEcgStatus: document.getElementById('dash-rpt-ecg-status'),
  dashRptRemark: document.getElementById('dash-rpt-remark'),
  dashViewReportBtn: document.getElementById('dash-view-report-btn'),
  dashDownloadReportBtn: document.getElementById('dash-download-report-btn'),
  dashViewFullHistoryBtn: document.getElementById('dash-view-full-history-btn'),
  qaAnalyzeEcg: document.getElementById('qa-analyze-ecg'),
  qaDownloadReport: document.getElementById('qa-download-report'),
  qaShareReport: document.getElementById('qa-share-report'),
  qaBookAppointment: document.getElementById('qa-book-appointment'),

  // Live Monitoring Page
  simStatusIndicator: document.getElementById('sim-status-indicator'),
  liveScenarioSelect: document.getElementById('live-scenario-select'),
  liveStreamToggleBtn: document.getElementById('live-stream-toggle-btn'),
  liveStreamBtnIcon: document.getElementById('live-stream-btn-icon'),
  liveStreamBtnText: document.getElementById('live-stream-btn-text'),
  liveHrNumber: document.getElementById('live-hr-number'),
  liveSpo2Number: document.getElementById('live-spo2-number'),
  liveTempNumber: document.getElementById('live-temp-number'),
  liveEcgRhythm: document.getElementById('live-ecg-rhythm'),
  liveEcgCanvas: document.getElementById('liveEcgCanvas'),

  // ECG Analysis Page
  sampleCountBadge: document.getElementById('sample-count-badge'),
  presetNormalBtn: document.getElementById('preset-normal-btn'),
  presetAbnormalBtn: document.getElementById('preset-abnormal-btn'),
  presetSyntheticBtn: document.getElementById('preset-synthetic-btn'),
  uploadDropzone: document.getElementById('upload-dropzone'),
  fileUploadInput: document.getElementById('file-upload-input'),
  rawSamplesTextarea: document.getElementById('raw-samples-textarea'),
  applyRawBtn: document.getElementById('apply-raw-btn'),
  analyzeEcgBtn: document.getElementById('analyze-ecg-btn'),
  analyzeIcon: document.getElementById('analyze-icon'),
  analyzeText: document.getElementById('analyze-text'),
  analyzeSpinner: document.getElementById('analyze-spinner'),
  beatCanvas: document.getElementById('beatCanvas'),
  waveformCaption: document.getElementById('waveform-caption'),
  waveformTooltip: document.getElementById('waveform-tooltip'),
  gainButtons: document.querySelectorAll('.btn-gain'),
  canvasPointCount: document.getElementById('canvas-point-count'),
  canvasMinAmp: document.getElementById('canvas-min-amp'),
  canvasMaxAmp: document.getElementById('canvas-max-amp'),
  canvasMeanAmp: document.getElementById('canvas-mean-amp'),

  // ECG Result Card
  resultCard: document.getElementById('result-card'),
  predictionBadge: document.getElementById('prediction-main-badge'),
  predictionIcon: document.getElementById('prediction-icon'),
  predictionText: document.getElementById('prediction-text'),
  predictionSummaryText: document.getElementById('prediction-summary-text'),
  resultStatusBadge: document.getElementById('result-status-badge'),
  probDisplay: document.getElementById('prob-pct-display'),
  probBar: document.getElementById('prob-bar-fill'),
  confDisplay: document.getElementById('confidence-pct-display'),
  confBar: document.getElementById('confidence-bar-fill'),
  classIdDisplay: document.getElementById('class-id-display'),
  classDescTag: document.getElementById('class-desc-tag'),

  // Medical Reports Page
  downloadReportPdfBtn: document.getElementById('download-report-pdf-btn'),
  shareReportBtn: document.getElementById('share-report-btn'),
  rptHrVal: document.getElementById('rpt-hr-val'),
  rptSpo2Val: document.getElementById('rpt-spo2-val'),
  rptTempVal: document.getElementById('rpt-temp-val'),
  rptEcgProb: document.getElementById('rpt-ecg-prob'),
  rptGeneratedTime: document.getElementById('rpt-generated-time'),

  // Health History Page
  historyDateFilter: document.getElementById('history-date-filter'),
  historyEcgFilter: document.getElementById('history-ecg-filter'),
  historySearchInput: document.getElementById('history-search-input'),
  fullHistoryTableBody: document.getElementById('full-history-table-body'),
  historyRecordsCount: document.getElementById('history-records-count'),
  exportHistoryCsvBtn: document.getElementById('export-history-csv-btn'),

  // Settings Page
  settingsNameInput: document.getElementById('settings-name-input'),
  saveProfileBtn: document.getElementById('save-profile-btn')
};

// Canvas Contexts
const beatCtx = DOM.beatCanvas ? DOM.beatCanvas.getContext('2d') : null;
const liveCtx = DOM.liveCanvas ? DOM.liveCanvas.getContext('2d') : null;


// =========================================================================
// 1. AUTHENTICATION & USER SESSION MANAGEMENT
// =========================================================================

async function checkAuthSession() {
  try {
    const res = await fetch('/api/auth/me');
    if (res.ok) {
      const data = await res.json();
      if (data.authenticated && data.user) {
        setAuthenticatedUser(data.user);
        return;
      }
    }
  } catch (err) {
    console.warn('Auth check error:', err);
  }

  // Not authenticated
  appState.isLoggedIn = false;
  appState.currentUser = null;
  DOM.authScreen.classList.remove('hidden');
  showLoginCard();
}

function setAuthenticatedUser(user) {
  appState.isLoggedIn = true;
  appState.currentUser = user;
  appState.patientName = user.full_name || 'Patient';
  appState.patientId = user.patient_id || 'PNT-2026-0001';

  // Update personalized UI text
  const firstName = appState.patientName.split(' ')[0] || appState.patientName;
  if (DOM.dashboardGreetingName) DOM.dashboardGreetingName.textContent = firstName;
  if (DOM.topbarUserName) DOM.topbarUserName.textContent = appState.patientName;
  if (DOM.sidebarPatientName) DOM.sidebarPatientName.textContent = appState.patientName;
  
  // Patient ID in sidebar
  const roleTag = document.querySelector('.patient-role-tag');
  if (roleTag) roleTag.textContent = `ID: ${appState.patientId}`;

  // Settings inputs
  if (DOM.settingsNameInput) DOM.settingsNameInput.value = appState.patientName;
  const settingsPidInput = document.querySelector('#settings-page input[readonly]');
  if (settingsPidInput) settingsPidInput.value = appState.patientId;

  // Medical report patient info
  const rptNameEl = document.querySelector('.report-patient-grid .patient-field:nth-child(1) strong');
  if (rptNameEl) rptNameEl.textContent = appState.patientName;
  const rptIdEl = document.querySelector('.report-patient-grid .patient-field:nth-child(2) strong');
  if (rptIdEl) rptIdEl.textContent = appState.patientId;

  // Hide auth modal and show dashboard
  DOM.authScreen.classList.add('hidden');
  navigateToPage('dashboard-page');
}

function showLoginCard() {
  DOM.loginCard.classList.remove('hidden');
  DOM.registerCard.classList.add('hidden');
  hideAlerts();
  lucide.createIcons();
}

function showRegisterCard() {
  DOM.registerCard.classList.remove('hidden');
  DOM.loginCard.classList.add('hidden');
  hideAlerts();
  lucide.createIcons();
}

function showLoginAlert(msg, isSuccess = false) {
  DOM.loginAlertText.textContent = msg;
  DOM.loginAlertBox.className = `auth-alert-box ${isSuccess ? 'success' : ''}`;
  DOM.loginAlertBox.classList.remove('hidden');
  lucide.createIcons();
}

function showRegisterAlert(msg, isSuccess = false) {
  DOM.registerAlertText.textContent = msg;
  DOM.registerAlertBox.className = `auth-alert-box ${isSuccess ? 'success' : ''}`;
  DOM.registerAlertBox.classList.remove('hidden');
  lucide.createIcons();
}

function hideAlerts() {
  if (DOM.loginAlertBox) DOM.loginAlertBox.classList.add('hidden');
  if (DOM.registerAlertBox) DOM.registerAlertBox.classList.add('hidden');
}

// 1.1 Login Submission
async function handleLoginSubmit(e) {
  if (e) e.preventDefault();
  hideAlerts();

  const email = DOM.loginEmailInput.value.trim();
  const password = DOM.loginPasswordInput.value;

  if (!email || !password) {
    showLoginAlert('Please enter both email address and password.');
    return;
  }

  DOM.loginSubmitBtn.disabled = true;
  DOM.loginSubmitBtn.innerHTML = '<span class="btn-spinner"></span> Authenticating...';

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    const data = await res.json();

    if (res.ok && data.status === 'SUCCESS' && data.user) {
      setAuthenticatedUser(data.user);
    } else {
      showLoginAlert(data.error || 'Invalid email or password.');
    }
  } catch (err) {
    showLoginAlert(`Connection error: ${err.message}`);
  } finally {
    DOM.loginSubmitBtn.disabled = false;
    DOM.loginSubmitBtn.innerHTML = '<i data-lucide="log-in" class="btn-icon"></i> <span>LOGIN</span>';
    lucide.createIcons();
  }
}

// 1.2 Registration Submission
async function handleRegisterSubmit(e) {
  if (e) e.preventDefault();
  hideAlerts();

  const fullName = DOM.regNameInput.value.trim();
  const email = DOM.regEmailInput.value.trim();
  const password = DOM.regPasswordInput.value;
  const confirmPassword = DOM.regConfirmPasswordInput.value;
  const phone = DOM.regPhoneInput ? DOM.regPhoneInput.value.trim() : null;
  const age = DOM.regAgeInput ? DOM.regAgeInput.value.trim() : null;
  const gender = DOM.regGenderSelect ? DOM.regGenderSelect.value : null;

  // Client Validation
  if (!fullName) {
    showRegisterAlert('Please enter your full name.');
    DOM.regNameInput.focus();
    return;
  }
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    showRegisterAlert('Please enter a valid email address.');
    DOM.regEmailInput.focus();
    return;
  }
  if (!password || password.length < 6) {
    showRegisterAlert('Password must be at least 6 characters long.');
    DOM.regPasswordInput.focus();
    return;
  }
  if (password !== confirmPassword) {
    showRegisterAlert('Passwords do not match.');
    DOM.regConfirmPasswordInput.focus();
    return;
  }

  DOM.registerSubmitBtn.disabled = true;
  DOM.registerSubmitBtn.innerHTML = '<span class="btn-spinner"></span> Creating Account...';

  try {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        full_name: fullName,
        email,
        password,
        confirm_password: confirmPassword,
        phone,
        age: age ? parseInt(age) : null,
        gender
      })
    });

    const data = await res.json();

    if (res.ok && data.status === 'SUCCESS') {
      // Clear registration form
      DOM.registerForm.reset();
      // Switch to login card with success banner and pre-filled email
      showLoginCard();
      DOM.loginEmailInput.value = email;
      DOM.loginPasswordInput.value = '';
      showLoginAlert('Account created successfully! Please login with your credentials.', true);
      DOM.loginPasswordInput.focus();
    } else {
      showRegisterAlert(data.error || 'Failed to create account.');
    }
  } catch (err) {
    showRegisterAlert(`Connection error: ${err.message}`);
  } finally {
    DOM.registerSubmitBtn.disabled = false;
    DOM.registerSubmitBtn.innerHTML = '<i data-lucide="check-circle" class="btn-icon"></i> <span>CREATE ACCOUNT</span>';
    lucide.createIcons();
  }
}

// 1.3 Logout Handler
async function handleLogout() {
  try {
    await fetch('/api/auth/logout', { method: 'POST' });
  } catch (err) {}

  appState.isLoggedIn = false;
  appState.currentUser = null;
  DOM.authScreen.classList.remove('hidden');
  showLoginCard();
  showLoginAlert('You have been logged out successfully.', true);
}


// =========================================================================
// 2. NAVIGATION & ROUTING
// =========================================================================

function navigateToPage(pageId) {
  if (!appState.isLoggedIn) {
    DOM.authScreen.classList.remove('hidden');
    showLoginCard();
    return;
  }

  appState.currentPage = pageId;

  // Update Nav items active state
  DOM.navButtons.forEach(btn => {
    if (btn.getAttribute('data-page') === pageId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Switch View Content
  DOM.pageViews.forEach(view => {
    if (view.id === pageId) {
      view.classList.add('active');
    } else {
      view.classList.remove('active');
    }
  });

  // Close mobile sidebar if open
  if (window.innerWidth <= 768) {
    DOM.appSidebar.classList.remove('open');
  }

  // Redraw canvases on tab change
  setTimeout(resizeCanvases, 50);

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}


// =========================================================================
// 3. API HEALTH & SERVER INTEGRATION
// =========================================================================

async function checkAPIHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      DOM.topbarApiDot.className = 'status-indicator-dot online';
      DOM.topbarApiText.textContent = 'API Connected';
      appState.apiConnected = true;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    DOM.topbarApiDot.className = 'status-indicator-dot offline';
    DOM.topbarApiText.textContent = 'API Offline';
    appState.apiConnected = false;
  }
}

async function loadSampleBeatsFromServer() {
  try {
    const res = await fetch('/api/ecg/sample-beats');
    if (res.ok) {
      const data = await res.json();
      BUILTIN_SAMPLES.normal = data.normal_beat.samples;
      BUILTIN_SAMPLES.abnormal = data.abnormal_beat.samples;
    }
  } catch (err) {
    console.warn('Could not fetch sample beats from API; generating fallback templates.');
  }

  if (!BUILTIN_SAMPLES.normal) BUILTIN_SAMPLES.normal = generateSyntheticBeat(false);
  if (!BUILTIN_SAMPLES.abnormal) BUILTIN_SAMPLES.abnormal = generateSyntheticBeat(true);

  // Load verified Normal beat (#0) as initial state
  loadSamples(BUILTIN_SAMPLES.normal, 'Known Normal Beat (#0 - Patient 100)');
}

function generateSyntheticBeat(isAbnormal = false) {
  const samples = [];
  for (let i = 0; i < 256; i++) {
    const t = i - 100;
    let v = 0.0;
    if (!isAbnormal) {
      v += 0.18 * Math.exp(-Math.pow(i - 65, 2) / 60);
      v -= 0.15 * Math.exp(-Math.pow(i - 92, 2) / 10);
      v += 1.45 * Math.exp(-Math.pow(t, 2) / 18);
      v -= 0.35 * Math.exp(-Math.pow(i - 108, 2) / 12);
      v += 0.38 * Math.exp(-Math.pow(i - 165, 2) / 140);
    } else {
      v += 1.85 * Math.exp(-Math.pow(t, 2) / 70);
      v -= 0.65 * Math.exp(-Math.pow(i - 125, 2) / 50);
      v -= 0.40 * Math.exp(-Math.pow(i - 175, 2) / 120);
    }
    v += (Math.random() - 0.5) * 0.012;
    samples.push(parseFloat(v.toFixed(5)));
  }
  return samples;
}


// =========================================================================
// 4. ECG BEAT LOADING & WAVEFORM CANVAS RENDERING
// =========================================================================

function loadSamples(samplesArray, label = 'ECG Beat Window') {
  if (!Array.isArray(samplesArray)) return;

  appState.currentSamples = samplesArray.map(Number);
  appState.currentLabel = label;

  // Validation updates
  DOM.sampleCountBadge.textContent = `${appState.currentSamples.length} / 256 samples`;
  DOM.canvasPointCount.textContent = `${appState.currentSamples.length} samples`;

  if (appState.currentSamples.length === 256) {
    DOM.sampleCountBadge.className = 'badge badge-cyan-pill';
    DOM.sampleCountBadge.textContent = '256 / 256 samples valid';
  } else {
    DOM.sampleCountBadge.className = 'badge badge-critical';
    DOM.sampleCountBadge.textContent = `${appState.currentSamples.length} / 256 invalid`;
  }

  DOM.rawSamplesTextarea.value = JSON.stringify(appState.currentSamples);
  DOM.waveformCaption.textContent = `${label} | 256 samples | Medical ECG Grid`;

  if (appState.currentSamples.length > 0) {
    const min = Math.min(...appState.currentSamples);
    const max = Math.max(...appState.currentSamples);
    const mean = appState.currentSamples.reduce((a, b) => a + b, 0) / appState.currentSamples.length;
    DOM.canvasMinAmp.textContent = `${min >= 0 ? '+' : ''}${min.toFixed(3)} mV`;
    DOM.canvasMaxAmp.textContent = `${max >= 0 ? '+' : ''}${max.toFixed(3)} mV`;
    DOM.canvasMeanAmp.textContent = `${mean.toFixed(3)} (Z-Score)`;
  }

  drawBeatWaveform();
}

function resizeCanvases() {
  if (DOM.beatCanvas && beatCtx) {
    const rect = DOM.beatCanvas.parentElement.getBoundingClientRect();
    DOM.beatCanvas.width = rect.width * window.devicePixelRatio;
    DOM.beatCanvas.height = rect.height * window.devicePixelRatio;
    beatCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    drawBeatWaveform();
  }

  if (DOM.liveCanvas && liveCtx) {
    const rect = DOM.liveCanvas.parentElement.getBoundingClientRect();
    DOM.liveCanvas.width = rect.width * window.devicePixelRatio;
    DOM.liveCanvas.height = rect.height * window.devicePixelRatio;
    liveCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
}

window.addEventListener('resize', resizeCanvases);

/**
 * Draws Medical ECG Grid & Signal Trace on White Medical Card
 */
function drawBeatWaveform() {
  if (!DOM.beatCanvas || !beatCtx) return;

  const width = DOM.beatCanvas.width / window.devicePixelRatio;
  const height = DOM.beatCanvas.height / window.devicePixelRatio;

  // Clear background
  beatCtx.fillStyle = '#FFFFFF';
  beatCtx.fillRect(0, 0, width, height);

  // 1. Draw Medical Standard ECG Grid (Pink/Reddish classic medical paper styling)
  const minor = 12;
  const major = 60;

  // Minor Grid
  beatCtx.lineWidth = 0.5;
  beatCtx.strokeStyle = 'rgba(239, 68, 68, 0.12)';
  beatCtx.beginPath();
  for (let x = 0; x < width; x += minor) {
    beatCtx.moveTo(x, 0); beatCtx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += minor) {
    beatCtx.moveTo(0, y); beatCtx.lineTo(width, y);
  }
  beatCtx.stroke();

  // Major Grid
  beatCtx.lineWidth = 1.0;
  beatCtx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
  beatCtx.beginPath();
  for (let x = 0; x < width; x += major) {
    beatCtx.moveTo(x, 0); beatCtx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += major) {
    beatCtx.moveTo(0, y); beatCtx.lineTo(width, y);
  }
  beatCtx.stroke();

  // Baseline Center Guide
  beatCtx.lineWidth = 1.0;
  beatCtx.strokeStyle = 'rgba(100, 116, 139, 0.2)';
  beatCtx.beginPath();
  beatCtx.moveTo(0, height / 2);
  beatCtx.lineTo(width, height / 2);
  beatCtx.stroke();

  // 2. Draw 256 ECG Points Signal Line
  const points = appState.currentSamples;
  if (!points || points.length === 0) return;

  const midY = height / 2;
  const scaleY = (height / 4.5) * appState.gain;
  const stepX = width / (points.length - 1 || 1);

  beatCtx.lineWidth = 2.2;
  beatCtx.strokeStyle = '#0284C7';
  beatCtx.lineCap = 'round';
  beatCtx.lineJoin = 'round';

  beatCtx.beginPath();
  points.forEach((val, i) => {
    const x = i * stepX;
    const y = midY - (val * scaleY);
    if (i === 0) beatCtx.moveTo(x, y);
    else beatCtx.lineTo(x, y);
  });
  beatCtx.stroke();

  // 3. Render Discrete Sample Point Dots
  beatCtx.fillStyle = '#0369A1';
  points.forEach((val, i) => {
    if (i % 4 === 0) {
      const x = i * stepX;
      const y = midY - (val * scaleY);
      beatCtx.beginPath();
      beatCtx.arc(x, y, 1.8, 0, Math.PI * 2);
      beatCtx.fill();
    }
  });
}


// =========================================================================
// 5. ANALYZE ECG INFERENCE (POST /api/ecg/predict)
// =========================================================================

async function analyzeCurrentECG() {
  if (appState.isAnalyzing) return;

  const samples = appState.currentSamples;

  // Client Validation
  if (!samples || samples.length !== 256) {
    alert(`The HealthSense 1D-CNN predictor requires exactly 256 samples. Current window has ${samples ? samples.length : 0} samples.`);
    return;
  }

  const hasInvalid = samples.some(v => typeof v !== 'number' || isNaN(v) || !isFinite(v));
  if (hasInvalid) {
    alert('ECG window contains invalid non-numeric or NaN/Inf values.');
    return;
  }

  // Loading State
  appState.isAnalyzing = true;
  DOM.analyzeEcgBtn.disabled = true;
  DOM.analyzeIcon.classList.add('hidden');
  DOM.analyzeSpinner.classList.remove('hidden');
  DOM.analyzeText.textContent = 'Running 1D-CNN Inference...';

  try {
    const response = await fetch('/api/ecg/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ecg: samples })
    });

    const result = await response.json();

    if (response.ok && result.status === 'SUCCESS') {
      result.timestamp = new Date().toLocaleTimeString();
      appState.lastPrediction = result;
      displayPredictionResult(result);
      updateDashboardAndReports(result);
    } else {
      displayErrorResult(result.error || 'Prediction failed');
    }
  } catch (err) {
    displayErrorResult(`Connection error: ${err.message}`);
  } finally {
    appState.isAnalyzing = false;
    DOM.analyzeEcgBtn.disabled = false;
    DOM.analyzeIcon.classList.remove('hidden');
    DOM.analyzeSpinner.classList.add('hidden');
    DOM.analyzeText.textContent = 'ANALYZE ECG';
    lucide.createIcons();
  }
}

function displayPredictionResult(result) {
  const isAbnormal = result.prediction === 'ABNORMAL' || result.class_id === 1;
  const probPct = (result.abnormal_probability * 100).toFixed(2);
  const confPct = (result.confidence * 100).toFixed(2);

  // Result Card Styling
  if (isAbnormal) {
    DOM.resultCard.className = 'medical-panel-card result-panel-card abnormal';
    DOM.predictionBadge.className = 'result-main-badge abnormal';
    DOM.predictionText.textContent = 'ABNORMAL';
    DOM.predictionIcon.setAttribute('data-lucide', 'alert-triangle');
    DOM.predictionSummaryText.textContent =
      'Arrhythmia rhythm characteristics detected. Waveform features exceed the 30% decision threshold.';
    DOM.probBar.className = 'progress-fill abnormal';
    DOM.classIdDisplay.textContent = '1 (ABNORMAL)';
    DOM.classDescTag.textContent = 'Class 1: Arrhythmia / PVC';
  } else {
    DOM.resultCard.className = 'medical-panel-card result-panel-card normal';
    DOM.predictionBadge.className = 'result-main-badge normal';
    DOM.predictionText.textContent = 'NORMAL';
    DOM.predictionIcon.setAttribute('data-lucide', 'shield-check');
    DOM.predictionSummaryText.textContent =
      'ECG beat waveform features match standard normal sinus rhythm characteristics. Abnormal probability is below the 30% threshold.';
    DOM.probBar.className = 'progress-fill normal';
    DOM.classIdDisplay.textContent = '0 (NORMAL)';
    DOM.classDescTag.textContent = 'Class 0: Normal Sinus';
  }

  DOM.resultStatusBadge.textContent = 'Analysis Complete (200 OK)';
  DOM.probDisplay.textContent = `${probPct}%`;
  DOM.probBar.style.width = `${Math.min(100, Math.max(1, probPct))}%`;
  DOM.confDisplay.textContent = `${confPct}%`;
  DOM.confBar.style.width = `${Math.min(100, Math.max(5, confPct))}%`;

  lucide.createIcons();

  if (window.innerWidth < 768) {
    DOM.resultCard.scrollIntoView({ behavior: 'smooth' });
  }
}

function displayErrorResult(errMsg) {
  DOM.resultCard.className = 'medical-panel-card result-panel-card abnormal';
  DOM.predictionBadge.className = 'result-main-badge abnormal';
  DOM.predictionText.textContent = 'ERROR';
  DOM.predictionIcon.setAttribute('data-lucide', 'alert-circle');
  DOM.predictionSummaryText.textContent = `Inference rejection: ${errMsg}`;
  DOM.resultStatusBadge.textContent = 'ERROR (400 Bad Request)';
  lucide.createIcons();
}

function updateDashboardAndReports(result) {
  const isAbnormal = result.prediction === 'ABNORMAL' || result.class_id === 1;
  const probPct = (result.abnormal_probability * 100).toFixed(2);

  // Update Dashboard Report Card
  DOM.dashRptEcgStatus.textContent = isAbnormal ? `ARRHYTHMIA (Prob: ${probPct}%)` : `NORMAL (Sinus Rhythm)`;
  DOM.dashRptEcgStatus.className = isAbnormal ? 'meta-value text-rose' : 'meta-value text-emerald';
  DOM.dashRptTime.textContent = result.timestamp;
  DOM.dashSyncTime.textContent = 'Just now';

  // Update Medical Report Page
  DOM.rptEcgProb.textContent = `${probPct}% (${result.prediction})`;
  DOM.rptGeneratedTime.textContent = `${new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}, ${result.timestamp}`;

  // Prepend to Recent Readings Table
  const newRow = document.createElement('tr');
  newRow.innerHTML = `
    <td><strong>Today, ${result.timestamp}</strong></td>
    <td>${appState.currentVitals.heartRate} bpm</td>
    <td>${appState.currentVitals.spo2} %</td>
    <td>${appState.currentVitals.temperature} °C</td>
    <td><span class="badge badge-sm ${isAbnormal ? 'badge-warning' : 'badge-success'}">${result.prediction} (${probPct}%)</span></td>
    <td><span class="badge badge-sm ${isAbnormal ? 'badge-warning' : 'badge-success'}">${isAbnormal ? 'Observed' : 'Normal'}</span></td>
    <td><button class="btn-table-action" onclick="window.viewReportModal(0)">View</button></td>
  `;
  const historyTbody = document.getElementById('dash-history-table-body');
  if (historyTbody && historyTbody.firstChild) {
    historyTbody.insertBefore(newRow, historyTbody.firstChild);
  }
}


// =========================================================================
// 6. LIVE TELEMETRY STREAMING (SSE /api/stream)
// =========================================================================

function initLiveTelemetryStream() {
  if (appState.eventSource) appState.eventSource.close();

  appState.eventSource = new EventSource('/api/stream');

  appState.eventSource.addEventListener('telemetry', (e) => {
    if (!appState.isStreaming) return;
    try {
      const data = JSON.parse(e.data);
      updateTelemetryUI(data);
    } catch (err) {
      console.error('Failed to parse SSE telemetry packet:', err);
    }
  });
}

function updateTelemetryUI(data) {
  if (!data) return;

  // 1. Ingest ECG samples
  if (data.ecg_points && Array.isArray(data.ecg_points)) {
    data.ecg_points.forEach(pt => appState.liveBuffer.push(pt));
    if (appState.liveBuffer.length > appState.maxLivePoints * 2) {
      appState.liveBuffer = appState.liveBuffer.slice(-appState.maxLivePoints);
    }
  } else if (data.waveform && data.waveform.filtered) {
    appState.liveBuffer = data.waveform.filtered;
  }

  // 2. Update Vitals with strict Real Hardware NULL checking
  if (data.heart_rate !== null && data.heart_rate !== undefined && data.heart_rate > 0) {
    const hr = Math.round(data.heart_rate);
    appState.currentVitals.heartRate = hr;
    if (DOM.liveHrNumber) DOM.liveHrNumber.textContent = hr;
    if (DOM.dashHrVal) DOM.dashHrVal.textContent = hr;
    if (DOM.rptHrVal) DOM.rptHrVal.textContent = `${hr} BPM`;
  } else {
    appState.currentVitals.heartRate = null;
    if (DOM.liveHrNumber) DOM.liveHrNumber.textContent = '--';
    if (DOM.dashHrVal) DOM.dashHrVal.textContent = '--';
    if (DOM.rptHrVal) DOM.rptHrVal.textContent = 'Sensor Not Connected';
  }

  if (data.spo2 !== null && data.spo2 !== undefined && data.spo2 > 0) {
    const spo2 = parseFloat(data.spo2).toFixed(1);
    appState.currentVitals.spo2 = spo2;
    if (DOM.liveSpo2Number) DOM.liveSpo2Number.textContent = spo2;
    if (DOM.dashSpo2Val) DOM.dashSpo2Val.textContent = Math.round(data.spo2);
    if (DOM.rptSpo2Val) DOM.rptSpo2Val.textContent = `${spo2}%`;
  } else {
    appState.currentVitals.spo2 = null;
    if (DOM.liveSpo2Number) DOM.liveSpo2Number.textContent = '--';
    if (DOM.dashSpo2Val) DOM.dashSpo2Val.textContent = '--';
    if (DOM.rptSpo2Val) DOM.rptSpo2Val.textContent = 'Sensor Not Connected';
  }

  if (data.temperature !== null && data.temperature !== undefined && data.temperature > 0) {
    const temp = parseFloat(data.temperature).toFixed(1);
    appState.currentVitals.temperature = temp;
    if (DOM.liveTempNumber) DOM.liveTempNumber.textContent = temp;
    if (DOM.dashTempVal) DOM.dashTempVal.textContent = temp;
    if (DOM.rptTempVal) DOM.rptTempVal.textContent = `${temp} °C`;
  } else {
    appState.currentVitals.temperature = null;
    if (DOM.liveTempNumber) DOM.liveTempNumber.textContent = '--';
    if (DOM.dashTempVal) DOM.dashTempVal.textContent = '--';
    if (DOM.rptTempVal) DOM.rptTempVal.textContent = 'Sensor Not Connected';
  }

  // 3. Leads off / ECG Sensor Status
  if (data.leads_off) {
    if (DOM.liveEcgRhythm) {
      DOM.liveEcgRhythm.textContent = 'ECG Sensor Not Connected / Leads Off';
      DOM.liveEcgRhythm.className = 'metric-status-title text-amber';
    }
  } else if (data.ml_prediction && DOM.liveEcgRhythm) {
    const isAb = data.ml_prediction.is_abnormal;
    DOM.liveEcgRhythm.textContent = isAb ? 'Arrhythmia / Ectopic' : 'Normal Sinus';
    DOM.liveEcgRhythm.className = isAb ? 'metric-status-title text-rose' : 'metric-status-title text-emerald';
  }
}

function drawLiveTrace() {
  if (!DOM.liveCanvas || !liveCtx) return;

  const width = DOM.liveCanvas.width / window.devicePixelRatio;
  const height = DOM.liveCanvas.height / window.devicePixelRatio;

  liveCtx.fillStyle = '#FFFFFF';
  liveCtx.fillRect(0, 0, width, height);

  // Medical grid
  const minor = 10;
  const major = 50;
  liveCtx.lineWidth = 0.5;
  liveCtx.strokeStyle = 'rgba(239, 68, 68, 0.12)';
  liveCtx.beginPath();
  for (let x = 0; x < width; x += minor) {
    liveCtx.moveTo(x, 0); liveCtx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += minor) {
    liveCtx.moveTo(0, y); liveCtx.lineTo(width, y);
  }
  liveCtx.stroke();

  // Waveform
  const pts = appState.liveBuffer;
  if (pts.length < 2) return;

  const midY = height / 2;
  const scaleY = (height / 5.0) * appState.gain;
  const stepX = width / (appState.maxLivePoints - 1);

  liveCtx.lineWidth = 2.0;
  liveCtx.strokeStyle = '#0284C7';
  liveCtx.beginPath();

  const startIdx = Math.max(0, pts.length - appState.maxLivePoints);
  const displayPts = pts.slice(startIdx);

  displayPts.forEach((val, i) => {
    const x = i * stepX;
    const y = midY - (val * scaleY);
    if (i === 0) liveCtx.moveTo(x, y);
    else liveCtx.lineTo(x, y);
  });
  liveCtx.stroke();
}


// =========================================================================
// 7. EVENT LISTENERS & INTERACTIVITY
// =========================================================================

function setupEventListeners() {
  // Navigation Sidebar
  DOM.navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const pageId = btn.getAttribute('data-page');
      if (pageId) navigateToPage(pageId);
    });
  });

  // Sidebar Toggle (Mobile)
  if (DOM.sidebarToggleBtn) {
    DOM.sidebarToggleBtn.addEventListener('click', () => {
      DOM.appSidebar.classList.toggle('open');
    });
  }

  // --- Auth View Switchers ---
  if (DOM.gotoRegisterBtn) {
    DOM.gotoRegisterBtn.addEventListener('click', showRegisterCard);
  }

  if (DOM.gotoLoginBtn) {
    DOM.gotoLoginBtn.addEventListener('click', showLoginCard);
  }

  // --- Auth Form Submissions ---
  if (DOM.loginSubmitBtn) {
    DOM.loginSubmitBtn.addEventListener('click', handleLoginSubmit);
  }
  if (DOM.loginForm) {
    DOM.loginForm.addEventListener('submit', handleLoginSubmit);
  }

  if (DOM.registerSubmitBtn) {
    DOM.registerSubmitBtn.addEventListener('click', handleRegisterSubmit);
  }
  if (DOM.registerForm) {
    DOM.registerForm.addEventListener('submit', handleRegisterSubmit);
  }

  // --- Password Visibility Toggles ---
  if (DOM.passwordToggleBtn) {
    DOM.passwordToggleBtn.addEventListener('click', () => {
      const type = DOM.loginPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      DOM.loginPasswordInput.setAttribute('type', type);
      DOM.passwordEyeIcon.setAttribute('data-lucide', type === 'password' ? 'eye' : 'eye-off');
      lucide.createIcons();
    });
  }

  if (DOM.regPasswordToggleBtn) {
    DOM.regPasswordToggleBtn.addEventListener('click', () => {
      const type = DOM.regPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      DOM.regPasswordInput.setAttribute('type', type);
      DOM.regPasswordEyeIcon.setAttribute('data-lucide', type === 'password' ? 'eye' : 'eye-off');
      lucide.createIcons();
    });
  }

  if (DOM.regConfirmPasswordToggleBtn) {
    DOM.regConfirmPasswordToggleBtn.addEventListener('click', () => {
      const type = DOM.regConfirmPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      DOM.regConfirmPasswordInput.setAttribute('type', type);
      DOM.regConfirmPasswordEyeIcon.setAttribute('data-lucide', type === 'password' ? 'eye' : 'eye-off');
      lucide.createIcons();
    });
  }

  // --- Forgot Password Link Notice ---
  if (DOM.forgotPasswordLink) {
    DOM.forgotPasswordLink.addEventListener('click', (e) => {
      e.preventDefault();
      showLoginAlert('Password recovery is not configured in this local prototype. Please use the pre-filled demo credentials or create a new account.');
    });
  }

  // --- Logout Buttons ---
  if (DOM.logoutBtn) {
    DOM.logoutBtn.addEventListener('click', handleLogout);
  }

  if (DOM.dropdownLogoutBtn) {
    DOM.dropdownLogoutBtn.addEventListener('click', () => {
      handleLogout();
      DOM.profileDropdownMenu.classList.add('hidden');
    });
  }

  // --- Dropdown Toggles ---
  if (DOM.notificationBtn) {
    DOM.notificationBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      DOM.notificationDropdown.classList.toggle('hidden');
      DOM.profileDropdownMenu.classList.add('hidden');
    });
  }

  if (DOM.userProfileToggle) {
    DOM.userProfileToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      DOM.profileDropdownMenu.classList.toggle('hidden');
      DOM.notificationDropdown.classList.add('hidden');
    });
  }

  document.addEventListener('click', () => {
    if (DOM.notificationDropdown) DOM.notificationDropdown.classList.add('hidden');
    if (DOM.profileDropdownMenu) DOM.profileDropdownMenu.classList.add('hidden');
  });

  if (DOM.dropdownSettingsBtn) {
    DOM.dropdownSettingsBtn.addEventListener('click', () => {
      navigateToPage('settings-page');
      DOM.profileDropdownMenu.classList.add('hidden');
    });
  }

  if (DOM.dropdownReportsBtn) {
    DOM.dropdownReportsBtn.addEventListener('click', () => {
      navigateToPage('medical-reports-page');
      DOM.profileDropdownMenu.classList.add('hidden');
    });
  }

  // Quick Action Buttons
  if (DOM.dashQuickAnalyzeBtn) DOM.dashQuickAnalyzeBtn.addEventListener('click', () => navigateToPage('ecg-analysis-page'));
  if (DOM.qaAnalyzeEcg) DOM.qaAnalyzeEcg.addEventListener('click', () => navigateToPage('ecg-analysis-page'));
  if (DOM.qaDownloadReport) DOM.qaDownloadReport.addEventListener('click', () => navigateToPage('medical-reports-page'));
  if (DOM.qaShareReport) DOM.qaShareReport.addEventListener('click', () => alert(`Report sharing link copied to clipboard! (Confidential Patient Record ${appState.patientId})`));
  if (DOM.qaBookAppointment) DOM.qaBookAppointment.addEventListener('click', () => alert('Appointment Booking modal: Clinician consultation scheduled for 28 Aug 2026. (Demo Mode)'));
  if (DOM.dashViewReportBtn) DOM.dashViewReportBtn.addEventListener('click', () => navigateToPage('medical-reports-page'));
  if (DOM.dashDownloadReportBtn) DOM.dashDownloadReportBtn.addEventListener('click', () => navigateToPage('medical-reports-page'));
  if (DOM.dashViewFullHistoryBtn) DOM.dashViewFullHistoryBtn.addEventListener('click', () => navigateToPage('health-history-page'));

  // ECG Preset Buttons
  if (DOM.presetNormalBtn) {
    DOM.presetNormalBtn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset-card').forEach(b => b.classList.remove('active'));
      DOM.presetNormalBtn.classList.add('active');
      loadSamples(BUILTIN_SAMPLES.normal, 'Known Normal Beat (#0 - Patient 100)');
    });
  }

  if (DOM.presetAbnormalBtn) {
    DOM.presetAbnormalBtn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset-card').forEach(b => b.classList.remove('active'));
      DOM.presetAbnormalBtn.classList.add('active');
      loadSamples(BUILTIN_SAMPLES.abnormal, 'Known Arrhythmia / PVC (#6077 - Patient 105)');
    });
  }

  if (DOM.presetSyntheticBtn) {
    DOM.presetSyntheticBtn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset-card').forEach(b => b.classList.remove('active'));
      DOM.presetSyntheticBtn.classList.add('active');
      const synth = generateSyntheticBeat(false);
      loadSamples(synth, 'Synthetic Sinus Rhythm Beat');
    });
  }

  // File Upload
  if (DOM.fileUploadInput) {
    DOM.fileUploadInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const text = evt.target.result.trim();
          let parsed;
          if (text.startsWith('[') || text.startsWith('{')) {
            parsed = JSON.parse(text);
            if (!Array.isArray(parsed) && parsed.ecg) parsed = parsed.ecg;
          } else {
            parsed = text.split(/[\s,]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
          }
          if (Array.isArray(parsed) && parsed.length > 0) {
            loadSamples(parsed, `Uploaded: ${file.name}`);
          }
        } catch (err) {
          alert(`Failed to parse file: ${err.message}`);
        }
      };
      reader.readAsText(file);
    });
  }

  // Apply Raw JSON
  if (DOM.applyRawBtn) {
    DOM.applyRawBtn.addEventListener('click', () => {
      try {
        const parsed = JSON.parse(DOM.rawSamplesTextarea.value);
        if (Array.isArray(parsed)) {
          loadSamples(parsed, 'Manual Raw ECG Samples');
        } else {
          alert('Input must be a JSON array of numbers.');
        }
      } catch (err) {
        alert(`Invalid JSON format: ${err.message}`);
      }
    });
  }

  // PDF Medical Report Generator
  if (DOM.downloadReportPdfBtn) {
    DOM.downloadReportPdfBtn.addEventListener('click', async () => {
      try {
        const payload = {
          patient_id: appState.patientId || 'PNT-2026-0001',
          patient_name: appState.patientName || 'Prajkta Patil',
          device_id: 'HS-ESP32-HARDWARE',
          heart_rate: appState.currentVitals.heartRate,
          spo2: appState.currentVitals.spo2,
          temperature: appState.currentVitals.temperature,
          ecg_result: appState.lastPrediction
        };

        const res = await fetch('/api/report/generate?format=html', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (res.ok) {
          const htmlText = await res.text();
          const printWindow = window.open('', '_blank');
          printWindow.document.write(htmlText);
          printWindow.document.close();
          setTimeout(() => {
            printWindow.print();
          }, 500);
        } else {
          alert('Failed to generate report.');
        }
      } catch (err) {
        alert(`Report generation error: ${err.message}`);
      }
    });
  }

  // Analyze Button
  if (DOM.analyzeEcgBtn) {
    DOM.analyzeEcgBtn.addEventListener('click', analyzeCurrentECG);
  }

  // Gain Buttons
  DOM.gainButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.gainButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      appState.gain = parseFloat(btn.dataset.gain) || 1.0;
      drawBeatWaveform();
    });
  });

  // Canvas Tooltip on Hover
  if (DOM.beatCanvas) {
    DOM.beatCanvas.addEventListener('mousemove', (e) => {
      const rect = DOM.beatCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const samples = appState.currentSamples;
      if (!samples || samples.length === 0) return;

      const idx = Math.max(0, Math.min(samples.length - 1, Math.round((x / rect.width) * (samples.length - 1))));
      const val = samples[idx];

      DOM.waveformTooltip.style.display = 'block';
      DOM.waveformTooltip.style.left = `${Math.min(rect.width - 130, Math.max(10, x + 15))}px`;
      DOM.waveformTooltip.style.top = `${Math.max(10, e.clientY - rect.top - 32)}px`;
      DOM.waveformTooltip.textContent = `Sample #${idx}: ${val >= 0 ? '+' : ''}${val.toFixed(3)} mV`;
    });

    DOM.beatCanvas.addEventListener('mouseleave', () => {
      DOM.waveformTooltip.style.display = 'none';
    });
  }

  // Live Scenario Selector
  if (DOM.liveScenarioSelect) {
    DOM.liveScenarioSelect.addEventListener('change', async (e) => {
      try {
        await fetch('/api/simulation/scenario', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: e.target.value })
        });
      } catch (err) {}
    });
  }

  // Live Stream Toggle
  if (DOM.liveStreamToggleBtn) {
    DOM.liveStreamToggleBtn.addEventListener('click', () => {
      appState.isStreaming = !appState.isStreaming;
      if (appState.isStreaming) {
        DOM.liveStreamBtnText.textContent = 'Pause Stream';
        DOM.liveStreamBtnIcon.setAttribute('data-lucide', 'pause');
      } else {
        DOM.liveStreamBtnText.textContent = 'Resume Stream';
        DOM.liveStreamBtnIcon.setAttribute('data-lucide', 'play');
      }
      lucide.createIcons();
    });
  }

  // History Filter & Search
  const filterHistory = () => {
    const query = (DOM.historySearchInput ? DOM.historySearchInput.value.toLowerCase() : '');
    const rhythmFilter = (DOM.historyEcgFilter ? DOM.historyEcgFilter.value : 'all');
    const rows = DOM.fullHistoryTableBody ? DOM.fullHistoryTableBody.querySelectorAll('tr') : [];
    let visibleCount = 0;

    rows.forEach(row => {
      const text = row.innerText.toLowerCase();
      const isNormal = text.includes('normal');
      const matchesSearch = text.includes(query);
      let matchesRhythm = true;
      if (rhythmFilter === 'normal') matchesRhythm = isNormal;
      if (rhythmFilter === 'abnormal') matchesRhythm = !isNormal;

      if (matchesSearch && matchesRhythm) {
        row.style.display = '';
        visibleCount++;
      } else {
        row.style.display = 'none';
      }
    });

    if (DOM.historyRecordsCount) {
      DOM.historyRecordsCount.textContent = `Showing ${visibleCount} Recorded Readings`;
    }
  };

  if (DOM.historySearchInput) DOM.historySearchInput.addEventListener('input', filterHistory);
  if (DOM.historyEcgFilter) DOM.historyEcgFilter.addEventListener('change', filterHistory);
  if (DOM.historyDateFilter) DOM.historyDateFilter.addEventListener('change', filterHistory);

  // Global Search Input
  if (DOM.globalSearchInput) {
    DOM.globalSearchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const val = DOM.globalSearchInput.value.trim().toLowerCase();
        if (val.includes('ecg') || val.includes('arrhythmia') || val.includes('rhythm')) {
          navigateToPage('ecg-analysis-page');
        } else if (val.includes('report')) {
          navigateToPage('medical-reports-page');
        } else if (val.includes('live') || val.includes('sensor') || val.includes('stream')) {
          navigateToPage('live-monitoring-page');
        } else if (val.includes('history') || val.includes('log')) {
          navigateToPage('health-history-page');
        } else {
          navigateToPage('dashboard-page');
        }
      }
    });
  }

  // Profile Save
  if (DOM.saveProfileBtn) {
    DOM.saveProfileBtn.addEventListener('click', () => {
      const name = DOM.settingsNameInput ? DOM.settingsNameInput.value.trim() : 'Prajkta Patil';
      appState.patientName = name;
      document.getElementById('sidebar-patient-name').textContent = name;
      document.getElementById('topbar-user-name').textContent = name;
      document.getElementById('dashboard-greeting-name').textContent = name.split(' ')[0] || name;
      alert('Patient Profile successfully updated!');
    });
  }
}

// Global modal helper for table rows
window.viewReportModal = function(idx) {
  navigateToPage('medical-reports-page');
};


// =========================================================================
// 8. RENDER LOOP & INITIALIZATION
// =========================================================================

function renderLoop() {
  if (appState.currentPage === 'live-monitoring-page') {
    drawLiveTrace();
  }
  requestAnimationFrame(renderLoop);
}

document.addEventListener('DOMContentLoaded', async () => {
  lucide.createIcons();
  setupEventListeners();
  resizeCanvases();

  // 1. Check API Health
  await checkAPIHealth();
  setInterval(checkAPIHealth, 8000);

  // 2. Check Active Session State
  await checkAuthSession();

  // 3. Load Sample Beats from Server
  await loadSampleBeatsFromServer();

  // 4. Connect Live Telemetry Stream
  initLiveTelemetryStream();

  // 5. Start 60 FPS Render Loop
  requestAnimationFrame(renderLoop);
});
