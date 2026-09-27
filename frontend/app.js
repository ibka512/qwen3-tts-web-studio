const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = {
  bootstrap: null, profiles: [], trash: [], history: [], mode: "clone",
  speaker: "Uncle_Fu", segmentPrefs: [], activeJob: null, preview: null,
  recorder: null, recordedFile: null, mediaStream: null, busy: false,
};
const MODE_LABEL = { clone: "我的音色", custom_voice: "官方音色" };
const STATUS_LABEL = { "待生成": "待生成", "进行中": "生成中", "已完成": "已完成", "失败": "失败" };
let jobPollTimer = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(data?.detail || ("请求失败（" + response.status + "）"));
  return data;
}
function postJSON(path, payload) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
}
function escapeHTML(value = "") {
  return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}
function showToast(message, error = false) {
  const toast = document.createElement("div");
  toast.className = "toast" + (error ? " error" : "");
  toast.textContent = message;
  $("#toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 3600);
}
function setBusy(value) {
  state.busy = value;
  if (value) $("#generate-button").disabled = true;
  else updateScriptStats();
  $("#design-button").disabled = value || !state.bootstrap?.models?.voice_design;
}
function languageOptions(selected = "Auto", includeAuto = true) {
  return (state.bootstrap?.languages || []).filter(item => includeAuto || item.value !== "Auto").map(item =>
    '<option value="' + escapeHTML(item.value) + '"' + (item.value === selected ? " selected" : "") + '>' + escapeHTML(item.label) + '</option>').join("");
}
function selectOptions(items, selected) {
  return items.map(item => '<option value="' + escapeHTML(item.value) + '"' + (item.value === selected ? " selected" : "") + '>' + escapeHTML(item.label) + '</option>').join("");
}
function switchView(name) {
  $$(".view").forEach(view => view.classList.toggle("active", view.id === "view-" + name));
  $$(".nav-item").forEach(button => {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "voices") refreshVoices();
  if (name === "history") refreshHistory();
}
function initTheme() {
  const saved = localStorage.getItem("qwen-tts-theme");
  const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $("#theme-toggle").setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
}
function toggleTheme() {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  localStorage.setItem("qwen-tts-theme", dark ? "dark" : "light");
  $("#theme-toggle").setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
}
function renderModelStatus() {
  const models = state.bootstrap?.models || {};
  const ready = Boolean(state.bootstrap && models.clone);
  const chip = $("#model-status");
  chip.classList.toggle("warning", !ready);
  chip.querySelector("span:last-child").textContent = ready ? "模型就绪" : "模型未就绪";
  $("#open-voice-design").disabled = !models.voice_design || state.busy;
  $("#design-button").disabled = !models.voice_design || state.busy;
  const missing = [];
  if (!models.clone) missing.push("Base");
  if (!models.custom_voice) missing.push("CustomVoice");
  if (!models.voice_design) missing.push("VoiceDesign");
  $("#model-note").textContent = missing.length ? "未检测到：" + missing.join("、") : "";
  $("#model-note").hidden = !missing.length;
}
function renderSpeakers(selected = state.speaker) {
  $("#speaker-grid").innerHTML = (state.bootstrap?.speakers || []).map(item => {
    const name = item.label.split(" · ")[0];
    const active = item.value === selected;
    return '<button type="button" class="speaker-card' + (active ? " active" : "") + '" data-speaker="' + escapeHTML(item.value) + '" aria-pressed="' + active + '">' +
      '<span class="speaker-name">' + escapeHTML(name) + '</span><span class="speaker-gender">' + escapeHTML(item.gender) + '</span></button>';
  }).join("");
}
function renderProfilePicker() {
  const picker = $("#profile-picker");
  const previous = picker.value;
  picker.innerHTML = state.profiles.map(item => '<option value="' + escapeHTML(item.id) + '">' + escapeHTML(item.name) + '</option>').join("");
  if (state.profiles.some(item => item.id === previous)) picker.value = previous;
  else if (state.profiles.length) picker.value = state.profiles[0].id;
  picker.disabled = !state.profiles.length;
  $("#no-profile-hint").hidden = state.profiles.length > 0;
  const profile = state.profiles.find(item => item.id === picker.value);
  const preview = $("#profile-preview");
  if (profile) { preview.src = profile.audio_url; preview.hidden = false; }
  else { preview.removeAttribute("src"); preview.hidden = true; }
  renderSegmentSettings();
}
function setMode(mode) {
  state.mode = mode;
  $$(".mode-choice").forEach(button => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  });
  $("#clone-controls").hidden = mode !== "clone";
  $("#custom-controls").hidden = mode !== "custom_voice";
  renderSegmentSettings();
}
function initLanguageControls() {
  $("#global-language").innerHTML = languageOptions("Auto");
  $("#design-language").innerHTML = languageOptions("Chinese", false);
  $("#global-instruction").value = state.bootstrap?.defaults?.instruction || "";
}
function updateScriptStats() {
  const lines = $("#script-input").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  const count = lines.reduce((sum, line) => sum + [...line].length, 0);
  $("#script-count").textContent = lines.length + " 段 · " + count.toLocaleString("zh-CN") + " 字";
  const invalid = lines.length > 8 || lines.some(line => [...line].length > 4000);
  $("#script-validation").textContent = lines.length > 8 ? "每批最多 8 段" :
    lines.some(line => [...line].length > 4000) ? "单段不能超过 4,000 字" : "最多 8 段，每段不超过 4,000 字";
  $("#script-validation").classList.toggle("invalid", invalid);
  $("#generate-button").disabled = state.busy || lines.length === 0 || invalid;
  const old = state.segmentPrefs;
  state.segmentPrefs = lines.map((_, index) => old[index] || {});
  renderSegmentSettings(lines);
}
function renderSegmentSettings(lines) {
  const source = lines || $("#script-input").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  const wrap = $("#segment-settings-list");
  $("#segment-setting-count").textContent = source.length ? source.length + " 段" : "使用整体设置";
  if (!source.length) {
    wrap.innerHTML = '<p class="inline-hint">输入文案后可逐段设置声音与语言。</p>';
    return;
  }
  const profiles = state.profiles.map(item => ({ value: item.id, label: item.name }));
  const speakers = state.bootstrap?.speakers || [];
  const voiceModes = [
    { value: "inherit", label: "使用整体声音" },
    { value: "clone", label: "我的音色" },
    { value: "custom_voice", label: "官方音色" },
  ];
  const langs = [{ value: "inherit", label: "使用整体语言" }, ...(state.bootstrap?.languages || [])];
  wrap.innerHTML = source.map((text, index) => {
    const pref = state.segmentPrefs[index] || {};
    const mode = pref.voice_mode || "inherit";
    const ownProfile = pref.profile_id || $("#profile-picker").value || "";
    const ownSpeaker = pref.speaker || state.speaker;
    const ownLanguage = pref.language || "inherit";
    let fields = '<label class="field-label wide">声音来源<select class="clay-input" data-segment="' + index + '" data-field="voice_mode">' + selectOptions(voiceModes, mode) + '</select></label>';
    if (mode === "clone") fields += '<label class="field-label wide">音色<select class="clay-input" data-segment="' + index + '" data-field="profile_id">' + selectOptions(profiles, ownProfile) + '</select></label>';
    if (mode === "custom_voice") {
      fields += '<label class="field-label wide">官方音色<select class="clay-input" data-segment="' + index + '" data-field="speaker">' + selectOptions(speakers, ownSpeaker) + '</select></label>';
      fields += '<label class="field-label wide">风格指令<textarea class="clay-input" rows="2" data-segment="' + index + '" data-field="instruction">' + escapeHTML(pref.instruction ?? $("#global-instruction").value) + '</textarea></label>';
    }
    fields += '<label class="field-label wide">语言<select class="clay-input" data-segment="' + index + '" data-field="language">' + selectOptions(langs, ownLanguage) + '</select></label>';
    return '<details class="segment-setting-card"><summary><span class="result-number">' + (index + 1) + '</span><span class="segment-excerpt">' + escapeHTML(text) + '</span></summary><div class="segment-fields">' + fields + '</div></details>';
  }).join("");
}
function getParams() {
  return {
    do_sample: $("#do-sample").checked,
    temperature: Number($("#temperature").value),
    top_p: Number($("#top-p").value),
    top_k: Number($("#top-k").value),
    repetition_penalty: Number($("#repetition").value),
    max_new_tokens: Number($("#max-tokens").value),
    subtalker_dosample: $("#subtalker-sample").checked,
    subtalker_temperature: Number($("#sub-temperature").value),
    subtalker_top_p: Number($("#sub-top-p").value),
    subtalker_top_k: Number($("#sub-top-k").value),
  };
}
function renderJob(record) {
  if (!record) { $("#active-generation").innerHTML = ""; return; }
  const complete = record.lines?.filter(line => line.status === "已完成").length || 0;
  const total = record.lines?.length || 0;
  const percent = Math.round((record.progress || 0) * 100);
  const text = record.state === "running" ? "生成中 · " + complete + "/" + total + " 段" :
    record.state === "completed" ? complete + "/" + total + " 段已完成" :
    record.state === "paused" ? "已暂停 · " + complete + "/" + total + " 段完成" :
    record.state === "partial" ? complete + "/" + total + " 段完成 · 部分失败" : "生成失败 · " + complete + "/" + total + " 段完成";
  let linesHTML = "";
  for (const line of record.lines || []) {
    linesHTML += '<article class="result-line"><span class="result-number">' + escapeHTML(line.index) + '</span>' +
      '<span class="result-text" title="' + escapeHTML(line.text) + '">' + escapeHTML(line.text) + '</span>' +
      '<span class="line-state' + (line.status === "已完成" ? " done" : "") + '">' + escapeHTML(STATUS_LABEL[line.status] || line.status) + '</span>';
    if (line.audio_url) linesHTML += '<audio controls preload="none" src="' + escapeHTML(line.audio_url) + '" aria-label="试听第 ' + line.index + ' 段"></audio><a class="download-link" href="' + escapeHTML(line.audio_url) + '" download>下载</a>';
    else if (line.error) linesHTML += '<span class="line-state">' + escapeHTML(line.error) + '</span>';
    linesHTML += '</article>';
  }
  const canResume = ["paused", "partial", "failed"].includes(record.state);
  let actions = '<a class="download-link" href="/api/history/' + encodeURIComponent(record.id) + '/download" download>下载整批</a>';
  if (canResume) actions += '<button class="button button-soft" type="button" data-resume="' + escapeHTML(record.id) + '">继续未完成</button>';
  $("#active-generation").innerHTML = '<div class="progress-card"><div class="progress-top"><span>' + escapeHTML(text) +
    '</span><span>' + percent + '%</span></div><div class="progress-track" role="progressbar" aria-valuenow="' + percent +
    '" aria-valuemin="0" aria-valuemax="100"><div class="progress-fill" style="width:' + percent +
    '%"></div></div><div class="result-actions">' + actions + '</div></div>' + linesHTML;
  $("#stop-job").hidden = record.state !== "running";
  $("#stop-job").disabled = record.state !== "running";
  if (record.state !== "running") { state.activeJob = null; setBusy(false); }
}
async function submitGeneration() {
  const lines = $("#script-input").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  if (!lines.length || lines.length > 8 || lines.some(line => [...line].length > 4000)) {
    $("#generation-message").textContent = "请检查文案分段。";
    return;
  }
  if (state.mode === "clone" && !$("#profile-picker").value) {
    $("#generation-message").textContent = "请先添加并选择一个音色。";
    return;
  }
  const segments = lines.map((text, index) => {
    const pref = state.segmentPrefs[index] || {};
    const mode = pref.voice_mode && pref.voice_mode !== "inherit" ? pref.voice_mode : state.mode;
    return {
      text,
      voice_mode: mode,
      profile_id: mode === "clone" ? (pref.profile_id || $("#profile-picker").value) : null,
      speaker: mode === "custom_voice" ? (pref.speaker || state.speaker) : null,
      instruction: mode === "custom_voice" ? (pref.instruction ?? $("#global-instruction").value.trim()) : "",
      language: pref.language && pref.language !== "inherit" ? pref.language : $("#global-language").value,
    };
  });
  const payload = {
    voice_mode: state.mode, profile_id: $("#profile-picker").value, speaker: state.speaker,
    instruction: $("#global-instruction").value.trim(), language: $("#global-language").value,
    segments, ...getParams(),
  };
  $("#generation-message").textContent = "";
  setBusy(true);
  try {
    const record = await postJSON("/api/jobs", payload);
    state.activeJob = record.id;
    renderJob(record);
    pollJob(record.id);
  } catch (error) {
    setBusy(false);
    $("#generation-message").textContent = error.message;
  }
}
async function pollJob(id) {
  if (jobPollTimer) clearTimeout(jobPollTimer);
  try {
    const record = await api("/api/jobs/" + encodeURIComponent(id));
    renderJob(record);
    if (record.state === "running") {
      setBusy(true);
      jobPollTimer = window.setTimeout(() => pollJob(id), 900);
    } else {
      setBusy(false);
      refreshHistory();
    }
  } catch (error) {
    setBusy(false);
    $("#generation-message").textContent = error.message;
  }
}
async function stopJob() {
  if (!state.activeJob) return;
  $("#stop-job").disabled = true;
  try {
    await postJSON("/api/jobs/" + encodeURIComponent(state.activeJob) + "/stop", {});
    showToast("当前段完成后暂停");
    pollJob(state.activeJob);
  } catch (error) { showToast(error.message, true); }
}
async function resumeJob(id) {
  setBusy(true);
  try {
    const record = await postJSON("/api/history/" + encodeURIComponent(id) + "/resume", {});
    switchView("creator");
    state.activeJob = record.id;
    renderJob(record);
    pollJob(record.id);
  } catch (error) { setBusy(false); showToast(error.message, true); }
}
function renderProfileCard(profile, trash = false) {
  const badge = profile.voice_design ? '<span class="badge design">设计音色</span>' :
    '<span class="badge">' + (profile.mode === "icl" ? "录音与原文" : "仅参考声音") + '</span>';
  const preview = profile.voice_design?.instruction ? '<p class="profile-meta">' + escapeHTML(profile.voice_design.instruction) + '</p>' : "";
  let actions;
  if (trash) actions = '<div class="profile-actions"><button class="button button-soft" data-restore="' + escapeHTML(profile.id) + '">恢复</button></div>';
  else actions = '<form class="rename-form" data-rename-form="' + escapeHTML(profile.id) + '" hidden><input class="clay-input" name="name" maxlength="36" value="' + escapeHTML(profile.name) + '" aria-label="音色名称"><button class="button button-primary" type="submit">保存名称</button></form>' +
    '<div class="profile-actions"><button class="button button-soft" data-use-profile="' + escapeHTML(profile.id) + '">用于创作</button><button class="button button-soft" data-toggle-rename="' + escapeHTML(profile.id) + '">改名</button><button class="button button-soft" data-export="' + escapeHTML(profile.id) + '">导出</button><button class="button button-soft" data-trash="' + escapeHTML(profile.id) + '">移入回收站</button></div>';
  return '<article class="profile-card" data-profile="' + escapeHTML(profile.id) + '"><div class="profile-card-top"><div><h3>' + escapeHTML(profile.name) +
    '</h3><span class="profile-meta">' + escapeHTML(profile.created_at || "") + '</span></div>' + badge + '</div>' + preview +
    '<audio controls preload="none" src="' + escapeHTML(profile.audio_url) + '" aria-label="试听 ' + escapeHTML(profile.name) + '"></audio>' + actions + '</article>';
}
async function refreshVoices() {
  try {
    const data = await api("/api/voices");
    state.profiles = data.profiles;
    state.trash = data.trash;
    renderProfilePicker();
    $("#profile-list").innerHTML = state.profiles.map(item => renderProfileCard(item)).join("");
    $("#profile-count").textContent = state.profiles.length + " 个";
    $("#profile-empty").hidden = state.profiles.length > 0;
    $("#trash-list").innerHTML = state.trash.map(item => renderProfileCard(item, true)).join("");
    $("#trash-count").textContent = state.trash.length + " 个";
    $("#trash-empty").hidden = state.trash.length > 0;
  } catch (error) { showToast(error.message, true); }
}
function renderHistoryCard(record) {
  const date = (record.created_at || "").replace("T", " ").slice(0, 16);
  const complete = record.lines?.filter(line => line.status === "已完成").length || 0;
  const canResume = ["paused", "partial", "failed"].includes(record.state);
  const stateText = record.state === "running" ? "生成中" : record.state === "completed" ? "已完成" :
    record.state === "paused" ? "已暂停" : record.state === "partial" ? "部分失败" : "失败";
  let actions = '<button class="button button-soft" data-load-settings="' + escapeHTML(record.id) + '">载入设置</button>';
  if (canResume) actions += '<button class="button button-soft" data-resume="' + escapeHTML(record.id) + '">继续未完成</button>';
  if (complete) actions += '<a class="download-link" href="/api/history/' + encodeURIComponent(record.id) + '/download" download>下载整批</a>';
  let lines = "";
  for (const line of record.lines || []) {
    lines += '<div class="history-line"><span class="result-number">' + escapeHTML(line.index) + '</span><span class="result-text">' + escapeHTML(line.text) +
      '</span><span class="line-state' + (line.status === "已完成" ? " done" : "") + '">' + escapeHTML(STATUS_LABEL[line.status] || line.status) + '</span>';
    if (line.audio_url) lines += '<audio controls preload="none" src="' + escapeHTML(line.audio_url) + '" aria-label="试听第 ' + line.index + ' 段"></audio><a class="download-link" href="' + escapeHTML(line.audio_url) + '" download>下载</a>';
    else if (line.error) lines += '<span class="line-state">' + escapeHTML(line.error) + '</span>';
    lines += '</div>';
  }
  return '<article class="history-card"><div class="history-card-header"><div><h2>' + escapeHTML(record.profile_name || "口播") + ' · ' + escapeHTML(date) +
    '</h2><span class="profile-meta">' + escapeHTML(MODE_LABEL[record.model_mode] || "多种音色") + ' · ' + complete + '/' + (record.lines?.length || 0) +
    ' 段完成</span></div><span class="badge">' + stateText + '</span></div><div class="history-card-actions">' + actions +
    '</div><div class="history-lines">' + lines + '</div></article>';
}
async function refreshHistory() {
  try {
    state.history = await api("/api/history");
    $("#history-list").innerHTML = state.history.map(renderHistoryCard).join("");
    $("#history-empty").hidden = state.history.length > 0;
  } catch (error) { showToast(error.message, true); }
}
async function loadSettings(id) {
  try {
    const settings = await api("/api/history/" + encodeURIComponent(id) + "/settings");
    setMode(settings.voice_mode === "custom_voice" ? "custom_voice" : "clone");
    if (settings.profile_id) $("#profile-picker").value = settings.profile_id;
    if (settings.speaker) { state.speaker = settings.speaker; renderSpeakers(); }
    $("#global-instruction").value = settings.instruction || "";
    $("#global-language").value = settings.params?.language || "Auto";
    $("#script-input").value = settings.script || "";
    $("#do-sample").checked = settings.params?.do_sample !== false;
    $("#subtalker-sample").checked = settings.params?.subtalker_dosample !== false;
    const sliderMap = [["temperature", "temperature"], ["top-p", "top_p"], ["top-k", "top_k"], ["repetition", "repetition_penalty"], ["max-tokens", "max_new_tokens"], ["sub-temperature", "subtalker_temperature"], ["sub-top-p", "subtalker_top_p"], ["sub-top-k", "subtalker_top_k"]];
    for (const pair of sliderMap) if (settings.params?.[pair[1]] != null) $("#" + pair[0]).value = settings.params[pair[1]];
    state.segmentPrefs = (settings.segments || []).map(segment => ({
      voice_mode: segment.mode, profile_id: segment.profile_id, speaker: segment.speaker,
      instruction: segment.instruction, language: segment.language,
    }));
    updateScriptStats();
    updateRangeValues();
    renderSegmentSettings();
    switchView("creator");
    showToast("设置已载入");
  } catch (error) { showToast(error.message, true); }
}
function updateRangeValues() {
  $("#temperature-value").value = Number($("#temperature").value).toFixed(2);
  $("#top-p-value").value = Number($("#top-p").value).toFixed(2);
  $("#top-k-value").value = $("#top-k").value;
  $("#repetition-value").value = Number($("#repetition").value).toFixed(2);
  $("#tokens-value").value = $("#max-tokens").value;
  $("#sub-temperature-value").value = Number($("#sub-temperature").value).toFixed(2);
  $("#sub-top-p-value").value = Number($("#sub-top-p").value).toFixed(2);
  $("#sub-top-k-value").value = $("#sub-top-k").value;
}
function updateCloneMode() { $("#transcript-field").hidden = $("#clone-mode").value !== "icl"; }
async function handleProfileForm(event) {
  event.preventDefault();
  const file = state.recordedFile || $("#profile-audio-file").files[0];
  if (!file) { showToast("请选择或录制一段参考音频", true); return; }
  const form = new FormData();
  form.append("name", $("#profile-name").value.trim());
  form.append("mode", $("#clone-mode").value);
  form.append("ref_text", $("#reference-text").value);
  form.append("audio", file, file.name || "recording.webm");
  const button = $("#profile-form button[type=submit]");
  button.disabled = true;
  button.textContent = "正在保存";
  try {
    const profile = await api("/api/voices", { method: "POST", body: form });
    state.recordedFile = null;
    $("#profile-audio-file").value = "";
    $("#profile-name").value = "";
    $("#reference-text").value = "";
    $("#record-preview").hidden = true;
    await refreshVoices();
    $("#profile-picker").value = profile.id;
    showToast("音色已保存");
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.textContent = "保存音色"; }
}
async function handleDesignForm(event) {
  event.preventDefault();
  const button = $("#design-button");
  button.disabled = true;
  button.textContent = "正在生成试听";
  setBusy(true);
  try {
    const preview = await postJSON("/api/voices/design/preview", {
      instruction: $("#design-instruction").value.trim(), text: $("#design-text").value.trim(),
      language: $("#design-language").value, ...getParams(),
    });
    state.preview = preview;
    $("#design-preview").innerHTML = '<audio controls autoplay src="' + escapeHTML(preview.audio_url) + '" aria-label="音色试听"></audio>' +
      '<div class="preview-save"><input class="clay-input" id="design-profile-name" maxlength="36" placeholder="音色名称" aria-label="音色名称"><button class="button button-primary" id="save-designed-voice" type="button">保存音色</button></div>';
    $("#design-preview").hidden = false;
    showToast("试听已生成");
  } catch (error) { showToast(error.message, true); }
  finally {
    button.disabled = false;
    button.textContent = "生成试听";
    setBusy(false);
    renderModelStatus();
  }
}
async function saveDesignedVoice() {
  if (!state.preview) return;
  try {
    const profile = await postJSON("/api/voices/design/save", { preview_id: state.preview.id, name: $("#design-profile-name").value.trim() });
    state.preview = null;
    $("#design-preview").hidden = true;
    $("#design-instruction").value = "";
    $("#design-text").value = "";
    await refreshVoices();
    $("#profile-picker").value = profile.id;
    switchView("creator");
    setMode("clone");
    showToast("设计音色已保存");
  } catch (error) { showToast(error.message, true); }
}
async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { showToast("此浏览器不支持录音", true); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaStream = stream;
    const mimeType = ["audio/webm;codecs=opus", "audio/mp4"].find(type => MediaRecorder.isTypeSupported(type));
    state.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    state.recorder.addEventListener("dataavailable", event => { if (event.data.size) chunks.push(event.data); });
    state.recorder.addEventListener("stop", () => {
      const blob = new Blob(chunks, { type: state.recorder.mimeType || "audio/webm" });
      const ext = blob.type.includes("mp4") ? "m4a" : "webm";
      state.recordedFile = new File([blob], "recording." + ext, { type: blob.type });
      $("#record-preview").innerHTML = '<audio controls src="' + URL.createObjectURL(blob) + '" aria-label="录音试听"></audio>';
      $("#record-preview").hidden = false;
      $("#record-status").textContent = "录音已完成";
      stream.getTracks().forEach(track => track.stop());
      state.mediaStream = null;
    }, { once: true });
    state.recorder.start();
    $("#record-start").hidden = true;
    $("#record-stop").hidden = false;
    $("#record-status").textContent = "正在录音";
  } catch (error) { showToast(error.message || "无法访问麦克风", true); }
}
function stopRecording() {
  if (state.recorder?.state === "recording") state.recorder.stop();
  $("#record-start").hidden = false;
  $("#record-stop").hidden = true;
}
async function handleProfileActions(event) {
  const button = event.target.closest("button");
  if (!button) return;
  const id = button.dataset.useProfile || button.dataset.trash || button.dataset.restore || button.dataset.export || button.dataset.toggleRename;
  if (!id) return;
  if (button.dataset.useProfile) {
    switchView("creator"); setMode("clone"); $("#profile-picker").value = id; renderProfilePicker();
  } else if (button.dataset.toggleRename) {
    const form = $('[data-rename-form="' + CSS.escape(id) + '"]');
    form.hidden = !form.hidden;
    if (!form.hidden) $("input", form).focus();
  } else if (button.dataset.trash) {
    if (!confirm("将此音色移入回收站？")) return;
    try { await postJSON("/api/voices/" + encodeURIComponent(id) + "/trash", {}); await refreshVoices(); showToast("音色已移入回收站"); }
    catch (error) { showToast(error.message, true); }
  } else if (button.dataset.restore) {
    try { await postJSON("/api/voices/" + encodeURIComponent(id) + "/restore", {}); await refreshVoices(); showToast("音色已恢复"); }
    catch (error) { showToast(error.message, true); }
  } else if (button.dataset.export) {
    try {
      const result = await postJSON("/api/voices/" + encodeURIComponent(id) + "/export", {});
      const link = document.createElement("a");
      link.href = result.download_url; link.download = result.filename; link.click();
    } catch (error) { showToast(error.message, true); }
  }
}
async function handleRename(event) {
  const form = event.target.closest("form[data-rename-form]");
  if (!form) return;
  event.preventDefault();
  try {
    await api("/api/voices/" + encodeURIComponent(form.dataset.renameForm), {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: new FormData(form).get("name") }),
    });
    await refreshVoices(); showToast("名称已保存");
  } catch (error) { showToast(error.message, true); }
}
function handleDelegatedActions(event) {
  const resume = event.target.closest("[data-resume]");
  if (resume) resumeJob(resume.dataset.resume);
  const load = event.target.closest("[data-load-settings]");
  if (load) loadSettings(load.dataset.loadSettings);
  const save = event.target.closest("#save-designed-voice");
  if (save) saveDesignedVoice();
}
function handleSegmentChange(event) {
  const field = event.target.dataset.field;
  const index = Number(event.target.dataset.segment);
  if (!field || !Number.isInteger(index)) return;
  const pref = state.segmentPrefs[index] || (state.segmentPrefs[index] = {});
  if (field === "voice_mode") {
    if (event.target.value === "inherit") {
      delete pref.voice_mode; delete pref.profile_id; delete pref.speaker; delete pref.instruction;
    } else pref.voice_mode = event.target.value;
    renderSegmentSettings();
  } else if (field === "language") {
    if (event.target.value === "inherit") delete pref.language;
    else pref.language = event.target.value;
  } else pref[field] = event.target.value;
}
async function init() {
  initTheme();
  try {
    state.bootstrap = await api("/api/bootstrap");
    initLanguageControls();
    renderSpeakers();
    renderModelStatus();
    const server = await api("/api/state");
    if (server.active_task) {
      setBusy(true);
      if (server.active_generation_id) {
        state.activeJob = server.active_generation_id;
        pollJob(server.active_generation_id);
      } else {
        $("#generation-message").textContent = "正在处理音色，请稍候";
        window.setTimeout(async () => {
          try {
            const current = await api("/api/state");
            if (!current.busy) {
              $("#generation-message").textContent = "";
              setBusy(false);
            }
          } catch (_) {}
        }, 1500);
      }
    }
    await Promise.all([refreshVoices(), refreshHistory()]);
  } catch (error) {
    $("#generation-message").textContent = error.message;
    $("#model-status").classList.add("warning");
    $("#model-status").querySelector("span:last-child").textContent = "连接失败";
  }
}
$$(".nav-item").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
$$("[data-go-view]").forEach(button => button.addEventListener("click", () => switchView(button.dataset.goView)));
$("#theme-toggle").addEventListener("click", toggleTheme);
$$(".mode-choice").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
$("#speaker-grid").addEventListener("click", event => {
  const button = event.target.closest("[data-speaker]");
  if (!button) return;
  state.speaker = button.dataset.speaker; renderSpeakers(); renderSegmentSettings();
});
$("#script-input").addEventListener("input", updateScriptStats);
$("#clear-script").addEventListener("click", () => { $("#script-input").value = ""; updateScriptStats(); $("#script-input").focus(); });
$("#profile-picker").addEventListener("change", renderProfilePicker);
$("#global-language").addEventListener("change", renderSegmentSettings);
$("#segment-settings-list").addEventListener("change", handleSegmentChange);
$("#segment-settings-list").addEventListener("input", handleSegmentChange);
$$(".advanced-grid input[type=range]").forEach(input => input.addEventListener("input", updateRangeValues));
$("#generate-button").addEventListener("click", submitGeneration);
$("#stop-job").addEventListener("click", stopJob);
$("#active-generation").addEventListener("click", handleDelegatedActions);
$("#design-form").addEventListener("submit", handleDesignForm);
$("#design-preview").addEventListener("click", handleDelegatedActions);
$("#profile-form").addEventListener("submit", handleProfileForm);
$("#clone-mode").addEventListener("change", updateCloneMode);
$("#profile-audio-file").addEventListener("change", () => {
  state.recordedFile = null; $("#record-preview").hidden = true;
  $("#record-status").textContent = $("#profile-audio-file").files.length ? "已选择音频" : "";
});
$("#record-start").addEventListener("click", startRecording);
$("#record-stop").addEventListener("click", stopRecording);
$("#profile-list").addEventListener("click", handleProfileActions);
$("#trash-list").addEventListener("click", handleProfileActions);
$("#profile-list").addEventListener("submit", handleRename);
$("#refresh-history").addEventListener("click", refreshHistory);
$("#history-list").addEventListener("click", handleDelegatedActions);
$("#open-voice-design").addEventListener("click", () => { switchView("voices"); $("#design-instruction").focus(); });
updateRangeValues();
updateScriptStats();
init();
