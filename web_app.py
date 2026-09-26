from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import threading
import uuid
import zipfile
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(os.environ.get("QWEN_TTS_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
MODEL_DIR = Path(
    os.environ.get("QWEN_TTS_MODEL_DIR", ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
).expanduser().resolve()
CUSTOMVOICE_MODEL_DIR = Path(
    os.environ.get("QWEN_TTS_CUSTOM_MODEL_DIR", ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice")
).expanduser().resolve()
VOICE_DIR = ROOT / "voices"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
HISTORY_DIR = ROOT / "history"
EXPORT_DIR = ROOT / "exports"
TRASH_DIR = VOICE_DIR / ".trash"
for directory in (VOICE_DIR, OUTPUT_DIR, LOG_DIR, HISTORY_DIR, EXPORT_DIR, TRASH_DIR, ROOT / "cache" / "gradio"):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "web-studio.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("qwen-voice-studio")

LANGUAGES = [
    ("自动判断", "Auto"),
    ("中文", "Chinese"),
    ("英语", "English"),
    ("日语", "Japanese"),
    ("韩语", "Korean"),
    ("德语", "German"),
    ("法语", "French"),
    ("俄语", "Russian"),
    ("葡萄牙语", "Portuguese"),
    ("西班牙语", "Spanish"),
    ("意大利语", "Italian"),
]
CLONE_MODES = [
    ("音频 + 逐字稿 · 模仿更完整", "icl"),
    ("仅提取音色 · 不需要逐字稿", "x_vector"),
]
CUSTOM_VOICE_CHOICES = [
    ("Uncle Fu · 男声", "Uncle_Fu"),
    ("Ryan · 男声", "Ryan"),
    ("Dylan · 男声", "Dylan"),
    ("Eric · 男声", "Eric"),
    ("Aiden · 男声", "Aiden"),
    ("Vivian · 女声", "Vivian"),
    ("Serena · 女声", "Serena"),
    ("Ono Anna · 女声", "Ono_Anna"),
    ("Sohee · 女声", "Sohee"),
]
CUSTOM_VOICE_LABELS = dict((speaker, label) for label, speaker in CUSTOM_VOICE_CHOICES)
DEFAULT_CUSTOM_VOICE_INSTRUCTION = "低沉、醇厚、沉稳的纪录片播音腔，语速适中，吐字清晰，情绪克制。"
ACCEPTED_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".webm"}
INFERENCE_LOCK = threading.RLock()
HISTORY_LOCK = threading.RLock()
STOP_REQUESTED = threading.Event()
PROMPT_CACHE: dict[str, Any] = {}

MODEL: Any = None
MODEL_KIND: str | None = None


def _ensure_model(kind: str):
    """Load one local checkpoint at a time to keep Apple Silicon memory use bounded."""
    global MODEL, MODEL_KIND
    if kind not in {"clone", "custom_voice"}:
        raise ValueError("不支持的音色模式。")
    if MODEL is not None and MODEL_KIND == kind:
        return MODEL

    model_dir = MODEL_DIR if kind == "clone" else CUSTOMVOICE_MODEL_DIR
    if not model_dir.is_dir() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"模型文件没有找到：{model_dir}")

    if MODEL is not None:
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        if MODEL_KIND == "clone":
            PROMPT_CACHE.clear()
        MODEL = None
        MODEL_KIND = None
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    label = "Base" if kind == "clone" else "CustomVoice"
    LOG.info("Loading Qwen3-TTS %s from %s", label, model_dir)
    MODEL = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="mps",
        dtype=torch.bfloat16,
        attn_implementation=None,
        local_files_only=True,
    )
    MODEL_KIND = kind
    LOG.info("%s model ready: device=mps, dtype=bfloat16", label)
    return MODEL


_ensure_model("clone")

CSS = r"""
:root {
  color-scheme: light dark;
  --canvas: light-dark(#f2f3ed, #111713);
  --surface: light-dark(#fffefa, #1b231e);
  --surface-soft: light-dark(#f7f8f3, #222c25);
  --ink: light-dark(#202923, #e7ede8);
  --muted: light-dark(#66736a, #b1beb4);
  --subtle: light-dark(#87938a, #91a095);
  --line: light-dark(#dfe4dc, #354239);
  --line-strong: light-dark(#cbd5cc, #46574a);
  --forest: light-dark(#315944, #9abca1);
  --forest-hover: light-dark(#264936, #afcbb4);
  --forest-soft: light-dark(#e8f0e9, #293a2e);
  --forest-ink: light-dark(#2e5a40, #c4d9c8);
  --copper: light-dark(#b96849, #e1a080);
  --copper-soft: light-dark(#f5ebe5, #392d27);
  --warm-ink: light-dark(#754a35, #ebc2a8);
  --primary-ink: light-dark(#fffefa, #172119);
  --focus: light-dark(rgba(185, 104, 73, .55), rgba(225, 160, 128, .68));
  --shadow-card: 0 6px 24px light-dark(rgba(31, 48, 36, .045), rgba(0, 0, 0, .18));
  --shadow-button: 0 7px 16px light-dark(rgba(39, 75, 53, .15), rgba(0, 0, 0, .22));
}
html, body, .gradio-container {
  color-scheme: inherit;
  background: var(--canvas) !important;
  color: var(--ink) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Hiragino Sans GB", sans-serif !important;
}
.gradio-container {
  --background-fill-primary: var(--canvas) !important;
  --background-fill-secondary: var(--surface-soft) !important;
  --block-background-fill: var(--surface) !important;
  --block-label-background-fill: var(--surface) !important;
  --input-background-fill: var(--surface) !important;
  --body-text-color: var(--ink) !important;
  --block-label-text-color: var(--ink) !important;
  --block-title-text-color: var(--ink) !important;
  --input-text-color: var(--ink) !important;
  --input-placeholder-color: var(--subtle) !important;
  --input-border-color: var(--line-strong) !important;
  --block-border-color: var(--line) !important;
  --border-color-primary: var(--line) !important;
  --button-secondary-background-fill: var(--surface-soft) !important;
  --button-secondary-text-color: var(--forest-ink) !important;
  --button-secondary-border-color: var(--line) !important;
  --button-primary-background-fill: var(--forest) !important;
  --button-primary-text-color: var(--primary-ink) !important;
  --checkbox-background-color: var(--surface) !important;
  width: min(100%, 1560px) !important;
  max-width: 1560px !important;
  padding: 24px clamp(16px, 3.2vw, 48px) 40px !important;
  margin: 0 auto !important;
}
footer, .built-with, .gradio-container > .main > .wrap > .contain > .footer { display: none !important; }
#studio-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  min-height: 94px;
  padding: 19px 24px;
  margin-bottom: 16px;
  border: 1px solid var(--line);
  border-radius: 17px;
  color: var(--ink);
  background: var(--surface);
  box-shadow: var(--shadow-card);
}
.header-copy { min-width: 0; }
.header-title {
  margin: 0;
  color: var(--ink);
  font-size: clamp(22px, 3vw, 29px);
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: -.04em;
  white-space: nowrap;
}
.header-side { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; }
.local-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 11px;
  border: 1px solid var(--line);
  border-radius: 99px;
  color: var(--forest-ink);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  background: var(--forest-soft);
}
.studio-grid { align-items: flex-start !important; gap: 16px !important; }
.panel {
  padding: 22px !important;
  border: 1px solid var(--line) !important;
  border-radius: 17px !important;
  background: var(--surface) !important;
  box-shadow: var(--shadow-card) !important;
}
.section-title {
  margin: 0 0 5px;
  color: var(--ink);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -.025em;
}
.section-note { color: var(--muted); font-size: 12px; line-height: 1.55; }
.soft-note {
  padding: 11px 12px;
  border: 1px solid var(--line);
  border-radius: 11px;
  background: var(--forest-soft);
  color: var(--forest-ink);
  font-size: 12px;
  line-height: 1.55;
}
.warm-note {
  padding: 11px 12px;
  border: 1px solid var(--line);
  border-radius: 11px;
  background: var(--copper-soft);
  color: var(--warm-ink);
  font-size: 12px;
  line-height: 1.55;
}
.status-box {
  min-height: 42px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 11px;
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}
#generate-button {
  min-height: 50px !important;
  border: 0 !important;
  border-radius: 12px !important;
  background: var(--forest) !important;
  color: var(--primary-ink) !important;
  font-weight: 700 !important;
  box-shadow: var(--shadow-button) !important;
  transition: transform .16s ease, background .16s ease, box-shadow .16s ease !important;
}
#generate-button:hover {
  background: var(--forest-hover) !important;
  transform: translateY(-1px);
}
#generate-button:active { transform: translateY(0); }
button.primary, .primary { border-radius: 11px !important; }
button.secondary, .secondary { border-radius: 10px !important; }
button, [role="button"] {
  cursor: pointer !important;
  transition: border-color .18s ease, background .18s ease, color .18s ease;
}
button:focus-visible, input:focus-visible, textarea:focus-visible, [tabindex="0"]:focus-visible {
  outline: 3px solid var(--focus) !important;
  outline-offset: 2px !important;
}
input, textarea { border-radius: 10px !important; }
textarea { line-height: 1.65 !important; }
label, .label { color: var(--ink) !important; font-weight: 600 !important; }
.gr-accordion {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--surface-soft) !important;
}
.accordion { border-radius: 12px !important; }
.audio-panel { border: 1px solid var(--line) !important; border-radius: 12px !important; overflow: hidden; }
.output-panel { padding-top: 8px; }
.output-title { margin-top: 4px; }
#speech-options { gap: 10px !important; }
#voice-cards fieldset { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 9px !important; }
#voice-cards fieldset > label {
  min-height: 48px !important;
  margin: 0 !important;
  padding: 12px 13px !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--surface-soft) !important;
  transition: border-color .16s ease, background .16s ease, transform .16s ease;
}
#voice-cards fieldset > label:hover { border-color: var(--line-strong) !important; transform: translateY(-1px); }
#voice-cards fieldset > label:has(input:checked) { border-color: var(--forest) !important; background: var(--forest-soft) !important; }
#first-use-guide { margin: 11px 0 13px; }
#line-table, #history-table { border-radius: 11px !important; overflow: hidden; }
#stop-button { min-height: 50px !important; border-radius: 12px !important; }
@media (max-width: 980px) {
  .studio-grid { flex-direction: column !important; }
  .studio-grid > * { width: 100% !important; min-width: 0 !important; }
}
@media (max-width: 640px) {
  .gradio-container { padding: 12px 12px 26px !important; }
  #studio-header { min-height: 72px; padding: 14px 15px; border-radius: 15px; gap: 10px; }
  .header-title { font-size: 22px; }
  .local-pill { padding: 7px 8px; font-size: 10px; }
  .panel { padding: 17px !important; border-radius: 15px !important; }
  #speech-options { flex-direction: column !important; align-items: stretch !important; gap: 8px !important; }
  #speech-options > * { width: 100% !important; min-width: 0 !important; flex: 1 1 100% !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
"""

HEADER_HTML = """
<div id="studio-header">
  <div class="header-copy">
    <h1 class="header-title">Qwen3-TTS</h1>
  </div>
  <div class="header-side">
    <span class="local-pill">双模式 · 1.7B</span>
  </div>
</div>
"""


def _read_profile(profile_dir: Path) -> dict[str, Any] | None:
    try:
        metadata = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
        audio_file = metadata.get("audio_file", "")
        if (
            metadata.get("id") != profile_dir.name
            or not isinstance(metadata.get("name"), str)
            or not metadata["name"].strip()
            or not isinstance(audio_file, str)
            or Path(audio_file).name != audio_file
            or not (profile_dir / audio_file).is_file()
        ):
            return None
        return metadata
    except (OSError, ValueError, TypeError):
        LOG.exception("Could not read voice profile %s", profile_dir)
        return None


def list_profiles() -> list[dict[str, Any]]:
    found = [metadata for folder in VOICE_DIR.iterdir() if folder.is_dir() and folder.name != ".trash"
             if (metadata := _read_profile(folder))]
    return sorted(found, key=lambda item: item.get("created_at", ""), reverse=True)


def list_trashed_profiles() -> list[dict[str, Any]]:
    found = [metadata for folder in TRASH_DIR.iterdir() if folder.is_dir()
             if (metadata := _read_profile(folder))]
    return sorted(found, key=lambda item: item.get("trashed_at", ""), reverse=True)


def profile_choices() -> list[tuple[str, str]]:
    return [
        (f"{item['name']} · {'ICL' if item.get('mode') == 'icl' else '音色向量'}", item["id"])
        for item in list_profiles()
    ]


def _profile_name_taken(name: str, profile_id: str | None = None) -> bool:
    profiles = list_profiles() + list_trashed_profiles()
    return any(item["id"] != profile_id and item["name"].casefold() == name.casefold() for item in profiles)


def trash_choices() -> list[tuple[str, str]]:
    return [(item["name"], item["id"]) for item in list_trashed_profiles()]


def _profile_dir(profile_id: str, allow_trash: bool = False) -> Path:
    if not isinstance(profile_id, str) or not re.fullmatch(r"[a-f0-9]{32}", profile_id):
        raise ValueError("音色档案编号无效。")
    active = VOICE_DIR / profile_id
    archived = TRASH_DIR / profile_id
    for candidate in ([active, archived] if allow_trash else [active]):
        resolved = candidate.resolve()
        if resolved.parent in {VOICE_DIR.resolve(), TRASH_DIR.resolve()} and resolved.is_dir():
            return resolved
    raise ValueError("这个音色档案已不存在，请刷新音色列表。")


def load_prompt(profile_id: str, allow_trash: bool = False):
    if not profile_id:
        raise ValueError("请先保存并选择一个音色档案。")
    profile_dir = _profile_dir(profile_id, allow_trash=allow_trash)
    metadata = _read_profile(profile_dir)
    if not metadata:
        raise ValueError("音色档案中的参考音频缺失或无法读取。")
    audio_path = (profile_dir / metadata["audio_file"]).resolve()
    if audio_path.parent != profile_dir:
        raise ValueError("档案中的参考音频路径无效。")
    with INFERENCE_LOCK:
        _ensure_model("clone")
        if profile_id in PROMPT_CACHE:
            return PROMPT_CACHE[profile_id]
        prompt_items = MODEL.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=(metadata.get("ref_text") or None),
            x_vector_only_mode=(metadata.get("mode") == "x_vector"),
        )
    PROMPT_CACHE[profile_id] = prompt_items
    return prompt_items


def _profile_audio_path(profile_id: str | None) -> str | None:
    if not profile_id:
        return None
    try:
        folder = _profile_dir(profile_id)
        metadata = _read_profile(folder)
        return str(folder / metadata["audio_file"]) if metadata else None
    except ValueError:
        return None


def load_profile_status(profile_id: str | None):
    if not profile_id:
        return "音色库为空。添加一段参考音频后，就可以开始生成。", None, ""
    try:
        load_prompt(profile_id)
        metadata = next((item for item in list_profiles() if item["id"] == profile_id), None)
        name = metadata["name"] if metadata else "已选音色"
        mode = "音频 + 逐字稿" if metadata and metadata.get("mode") == "icl" else "仅提取音色"
        return f"**{html.escape(name)} 已就绪** · {mode}", _profile_audio_path(profile_id), name
    except Exception as exc:
        LOG.exception("Failed to load voice profile %s", profile_id)
        return f"**音色载入失败：** {html.escape(str(exc))}", _profile_audio_path(profile_id), ""


def _profile_refresh_payload(profile_id: str | None):
    choices = profile_choices()
    valid_ids = {value for _, value in choices}
    selected = profile_id if profile_id in valid_ids else (choices[0][1] if choices else None)
    status, audio, name = load_profile_status(selected)
    return (
        gr.update(choices=choices, value=selected), status, audio, name,
        gr.update(visible=not choices), gr.update(choices=trash_choices()),
    )


def initialize_page(profile_id: str | None):
    return (*_profile_refresh_payload(profile_id), gr.update(choices=history_choices()))


def refresh_profiles(profile_id: str | None):
    payload = _profile_refresh_payload(profile_id)
    choices = profile_choices()
    valid_ids = {value for _, value in choices}
    selected = profile_id if profile_id in valid_ids else (choices[0][1] if choices else None)
    if selected:
        return payload
    return (payload[0], "音色列表已更新。添加参考音频后，可在这里选择音色。", None, "", payload[4], payload[5])


def save_voice_profile(name: str, audio_file: str, ref_text: str, mode: str, progress=gr.Progress()):
    clean_name = " ".join((name or "").split())
    if not clean_name:
        return (gr.update(choices=profile_choices()), "请为音色命名。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))
    if len(clean_name) > 36:
        return (gr.update(choices=profile_choices()), "名称请控制在 36 个字符以内。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))
    if _profile_name_taken(clean_name):
        return (gr.update(choices=profile_choices()), "这个名称已存在，请换一个。", None, "", gr.update(visible=False), gr.update(choices=trash_choices()))
    if not audio_file or not Path(audio_file).is_file():
        return (gr.update(choices=profile_choices()), "请上传或录制一段参考音频。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))
    transcript = (ref_text or "").strip()
    if mode == "icl" and not transcript:
        return (gr.update(choices=profile_choices()), "ICL 模式需要填写参考音频的逐字稿。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))

    source = Path(audio_file)
    suffix = source.suffix.lower()
    if suffix not in ACCEPTED_AUDIO:
        return (gr.update(choices=profile_choices()), "请使用 WAV、FLAC、MP3、M4A、OGG 等常见音频。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))
    if source.stat().st_size > 100 * 1024 * 1024:
        return (gr.update(choices=profile_choices()), "参考音频超过 100 MB，请先裁剪。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))

    profile_id = uuid.uuid4().hex
    profile_dir = VOICE_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=False)
    saved_audio = profile_dir / f"reference{suffix}"
    metadata = {
        "schema_version": 1, "id": profile_id, "name": clean_name,
        "audio_file": saved_audio.name, "mode": mode,
        "ref_text": transcript if mode == "icl" else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        shutil.copyfile(source, saved_audio)
        progress(0.15, desc="正在载入音色")
        with INFERENCE_LOCK:
            _ensure_model("clone")
            prompt_items = MODEL.create_voice_clone_prompt(
                ref_audio=str(saved_audio), ref_text=metadata["ref_text"] or None,
                x_vector_only_mode=(mode == "x_vector"),
            )
        (profile_dir / "profile.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        PROMPT_CACHE[profile_id] = prompt_items
        LOG.info("Saved voice profile %s (%s)", profile_id, clean_name)
        return (
            gr.update(choices=profile_choices(), value=profile_id),
            f"**{html.escape(clean_name)} 已保存并载入。**",
            str(saved_audio), clean_name, gr.update(visible=False), gr.update(choices=trash_choices()),
        )
    except Exception as exc:
        LOG.exception("Could not create voice profile %s", clean_name)
        PROMPT_CACHE.pop(profile_id, None)
        shutil.rmtree(profile_dir, ignore_errors=True)
        return (gr.update(choices=profile_choices()), f"音色保存失败：{html.escape(str(exc))}", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))


def rename_voice_profile(profile_id: str, name: str):
    clean_name = " ".join((name or "").split())
    choices = profile_choices()
    if not profile_id:
        return (gr.update(choices=choices), "先选择一个音色。", None, "", gr.update(visible=not choices), gr.update(choices=trash_choices()))
    if not clean_name or len(clean_name) > 36:
        return (gr.update(choices=choices), "名称需为 1 至 36 个字符。", _profile_audio_path(profile_id), clean_name, gr.update(visible=not choices), gr.update(choices=trash_choices()))
    if _profile_name_taken(clean_name, profile_id=profile_id):
        return (gr.update(choices=choices), "这个名称已被使用。", _profile_audio_path(profile_id), clean_name, gr.update(visible=False), gr.update(choices=trash_choices()))
    try:
        folder = _profile_dir(profile_id)
        metadata = _read_profile(folder)
        if not metadata:
            raise ValueError("音色档案无法读取。")
        metadata["name"] = clean_name
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        (folder / "profile.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return (gr.update(choices=profile_choices(), value=profile_id), f"音色已改名为 **{html.escape(clean_name)}**。", _profile_audio_path(profile_id), clean_name, gr.update(visible=False), gr.update(choices=trash_choices()))
    except Exception as exc:
        return (gr.update(choices=profile_choices()), f"改名失败：{html.escape(str(exc))}", _profile_audio_path(profile_id), clean_name, gr.update(visible=False), gr.update(choices=trash_choices()))


def move_profile_to_trash(profile_id: str):
    choices = profile_choices()
    try:
        folder = _profile_dir(profile_id)
        metadata = _read_profile(folder)
        if not metadata:
            raise ValueError("音色档案无法读取。")
        metadata["trashed_at"] = datetime.now().isoformat(timespec="seconds")
        (folder / "profile.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        destination = TRASH_DIR / profile_id
        shutil.move(str(folder), str(destination))
        PROMPT_CACHE.pop(profile_id, None)
        next_id = next((value for _, value in profile_choices()), None)
        status = f"**{html.escape(metadata['name'])}** 已移入回收站，可随时恢复。"
        return (gr.update(choices=profile_choices(), value=next_id), status, _profile_audio_path(next_id),
                next((item["name"] for item in list_profiles() if item["id"] == next_id), ""),
                gr.update(visible=not profile_choices()), gr.update(choices=trash_choices(), value=profile_id))
    except Exception as exc:
        return (gr.update(choices=choices), f"无法移入回收站：{html.escape(str(exc))}", _profile_audio_path(profile_id), "", gr.update(visible=not choices), gr.update(choices=trash_choices()))


def restore_profile(profile_id: str):
    try:
        folder = _profile_dir(profile_id, allow_trash=True)
        if folder.parent != TRASH_DIR.resolve():
            raise ValueError("此音色不在回收站中。")
        metadata = _read_profile(folder)
        if not metadata:
            raise ValueError("音色档案无法读取。")
        del metadata["trashed_at"]
        metadata["restored_at"] = datetime.now().isoformat(timespec="seconds")
        (folder / "profile.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        destination = VOICE_DIR / profile_id
        shutil.move(str(folder), str(destination))
        status = f"**{html.escape(metadata['name'])}** 已恢复。"
        return (gr.update(choices=profile_choices(), value=profile_id), status, _profile_audio_path(profile_id), metadata["name"],
                gr.update(visible=False), gr.update(choices=trash_choices(), value=None))
    except Exception as exc:
        return (gr.update(choices=profile_choices()), f"恢复失败：{html.escape(str(exc))}", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))


def export_voice_profile(profile_id: str):
    try:
        folder = _profile_dir(profile_id)
        metadata = _read_profile(folder)
        if not metadata:
            raise ValueError("音色档案无法读取。")
        safe_name = re.sub(r"[^\w\-]+", "_", metadata["name"], flags=re.UNICODE).strip("_")[:36] or "voice"
        zip_path = EXPORT_DIR / f"{safe_name}_{profile_id[:8]}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / "profile.json", "profile.json")
            archive.write(folder / metadata["audio_file"], metadata["audio_file"])
        return gr.update(value=str(zip_path), visible=True), f"**{html.escape(metadata['name'])}** 的档案已打包。"
    except Exception as exc:
        return gr.update(value=None, visible=False), f"导出失败：{html.escape(str(exc))}"


def update_script_stats(script: str):
    lines = [line.strip() for line in (script or "").splitlines() if line.strip()]
    too_long = sum(len(line) > 4000 for line in lines)
    if not lines:
        return "输入文本后，每一行会生成一段语音。", gr.update(interactive=False)
    if len(lines) > 8:
        return f"⚠️ 当前 **{len(lines)} 行**；每批最多 8 行。", gr.update(interactive=False)
    if too_long:
        return f"⚠️ 有 **{too_long} 行**超过 4,000 字符，请拆分后再生成。", gr.update(interactive=False)
    count = sum(len(line) for line in lines)
    return f"{len(lines)} 段 · {count:,} 个字符", gr.update(interactive=True)


def apply_sampling_preset(preset: str):
    presets = {
        "default": (True, 0.9, 1.0, 50, 1.05, 2048),
        "lower_randomness": (True, 0.65, 0.85, 40, 1.05, 2048),
        "higher_randomness": (True, 1.15, 1.0, 50, 1.05, 2048),
    }
    return presets.get(preset, presets["default"])


def _manifest_path(batch_id: str) -> Path:
    if not isinstance(batch_id, str) or not re.fullmatch(r"[a-f0-9]{32}", batch_id):
        raise ValueError("生成记录编号无效。")
    return HISTORY_DIR / f"{batch_id}.json"


def _write_history(record: dict[str, Any]):
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path = _manifest_path(record["id"])
    temp_path = path.with_suffix(".json.tmp")
    with HISTORY_LOCK:
        temp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)


def _read_history(batch_id: str) -> dict[str, Any]:
    record = json.loads(_manifest_path(batch_id).read_text(encoding="utf-8"))
    if record.get("id") != batch_id or not isinstance(record.get("lines"), list):
        raise ValueError("生成记录格式无效。")
    return record


def history_choices() -> list[tuple[str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in HISTORY_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not re.fullmatch(r"[a-f0-9]{32}", str(record.get("id", ""))) or path.name != f"{record['id']}.json":
                continue
            created = str(record.get("created_at", ""))[:16].replace("T", " ")
            name = str(record.get("profile_name", "音色"))[:20]
            count = len(record.get("lines", []))
            found.append((str(record.get("created_at", "")), f"{created} · {name} · {count} 段", record["id"]))
        except (OSError, ValueError, TypeError, KeyError):
            LOG.exception("Could not list generation history %s", path)
    return [(label, batch_id) for _, label, batch_id in sorted(found, reverse=True)]


def recover_interrupted_history():
    for path in HISTORY_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            batch_id = str(record.get("id", ""))
            if path.name != f"{batch_id}.json" or not re.fullmatch(r"[a-f0-9]{32}", batch_id):
                continue
            changed = False
            for line in record.get("lines", []):
                if line.get("status") != "进行中":
                    continue
                output_name = f"{batch_id}_{int(line['index']):02d}.wav"
                if _safe_output_path(output_name):
                    line["status"] = "已完成"
                    line["output_file"] = output_name
                    line["error"] = ""
                else:
                    line["status"] = "待生成"
                    line["error"] = ""
                changed = True
            if changed:
                _write_history(record)
        except (OSError, ValueError, TypeError, KeyError):
            LOG.exception("Could not recover interrupted generation %s", path)


def _safe_output_path(file_name: str | None) -> str | None:
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        return None
    path = (OUTPUT_DIR / file_name).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        return None
    return str(path)


def _record_files(record: dict[str, Any]) -> list[str]:
    return [path for line in record["lines"] if line.get("status") == "已完成"
            if (path := _safe_output_path(line.get("output_file")))]


def _history_rows(record: dict[str, Any]) -> list[list[str]]:
    return [[str(item.get("index", "")), str(item.get("text", "")), str(item.get("status", "待生成")),
             str(item.get("output_file") or ""), str(item.get("error") or "")]
            for item in record["lines"]]


def load_history_view(batch_id: str | None):
    if not batch_id:
        return "选择一条生成记录查看试听和逐段状态。", None, [], []
    try:
        record = _read_history(batch_id)
        files = _record_files(record)
        completed = sum(line.get("status") == "已完成" for line in record["lines"])
        status = f"**{html.escape(record.get('profile_name', '音色'))}** · {completed}/{len(record['lines'])} 段完成 · {str(record.get('created_at', ''))[:16].replace('T', ' ')}"
        return status, (files[0] if files else None), files, _history_rows(record)
    except Exception as exc:
        return f"读取记录失败：{html.escape(str(exc))}", None, [], []


def load_history_settings(batch_id: str | None):
    try:
        record = _read_history(batch_id)
        params = record["params"]
        choices = profile_choices()
        available_ids = {value for _, value in choices}
        mode = record.get("model_mode", "clone")
        if mode not in {"clone", "custom_voice"}:
            mode = "clone"
        profile_id = record.get("profile_id")
        profile_is_available = profile_id in available_ids
        selected_id = profile_id if profile_is_available else (choices[0][1] if choices else None)
        if mode == "custom_voice":
            speaker = record.get("speaker")
            if speaker not in CUSTOM_VOICE_LABELS:
                speaker = "Uncle_Fu"
            status = f"已载入官方音色 **{html.escape(CUSTOM_VOICE_LABELS[speaker])}** 的设置。可回到工作台修改文本后生成。"
        else:
            status = (
                f"已载入 **{html.escape(record.get('profile_name', '音色'))}** 的设置。可回到工作台修改文本后生成。"
                if profile_is_available else "这批记录使用的音色当前在回收站或已不可用；请先恢复音色，再回到工作台生成。"
            )
        return (
            gr.update(value=mode),
            gr.update(visible=(mode == "clone")),
            gr.update(visible=(mode == "custom_voice")),
            gr.update(choices=choices, value=selected_id),
            gr.update(value=speaker if mode == "custom_voice" else "Uncle_Fu"),
            params.get("instruction", DEFAULT_CUSTOM_VOICE_INSTRUCTION),
            "\n".join(item["text"] for item in record["lines"]),
            gr.update(value=params.get("language", "Auto")), bool(params.get("do_sample", True)),
            params.get("temperature", 0.9), params.get("top_p", 1.0), params.get("top_k", 50),
            params.get("repetition_penalty", 1.05), params.get("max_new_tokens", 2048),
            status,
        )
    except Exception as exc:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), DEFAULT_CUSTOM_VOICE_INSTRUCTION,
            "", gr.update(), True, 0.9, 1.0, 50, 1.05, 2048,
            f"无法载入设置：{html.escape(str(exc))}",
        )


def _visible_result(record: dict[str, Any]):
    files = _record_files(record)
    return files[0] if files else None, files


def _progress_state(record: dict[str, Any]):
    return {"batch_id": record["id"], "indexes": [item["index"] for item in record["lines"] if item.get("status") != "已完成"]}


def _retry_button_update(state: dict[str, Any] | None, running: bool = False):
    return gr.update(visible=bool(not running and state and state.get("indexes")))


def _batch_message(record: dict[str, Any], stopped: bool = False):
    lines = record["lines"]
    completed = sum(item.get("status") == "已完成" for item in lines)
    failed = sum(item.get("status") == "失败" for item in lines)
    pending = sum(item.get("status") == "待生成" for item in lines)
    if stopped:
        return f"已停止 · {completed}/{len(lines)} 段完成。已有音频已保存，可继续未完成部分。"
    if failed or pending:
        return f"完成 {completed}/{len(lines)} 段 · 失败 {failed} 段 · 待生成 {pending} 段。可重试或继续未完成部分。"
    return f"已完成 **{completed} 段**。所有 WAV 已保存到外接盘。"


def _stream_batch(record: dict[str, Any], indexes: list[int], progress):
    STOP_REQUESTED.clear()
    batch_id = record["id"]
    rows = _history_rows(record)
    audio, files = _visible_result(record)
    state = _progress_state(record)
    voice_mode = record.get("model_mode", "clone")
    if voice_mode not in {"clone", "custom_voice"}:
        voice_mode = "clone"
    yield audio, files, "正在载入音色…", rows, state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=True), _retry_button_update(state, running=True)
    try:
        with INFERENCE_LOCK:
            _ensure_model(voice_mode)
            if voice_mode == "clone":
                prompt_items = load_prompt(record.get("profile_id"), allow_trash=True)
            else:
                prompt_items = None
                if record.get("speaker") not in CUSTOM_VOICE_LABELS:
                    raise ValueError("这条记录中的官方音色无效，请从列表重新选择。")
    except Exception as exc:
        LOG.exception("Could not load model or voice for history batch %s", batch_id)
        for item in record["lines"]:
            if item["index"] in indexes and item.get("status") != "已完成":
                item["status"] = "失败"
                item["error"] = str(exc)[:500]
        _write_history(record)
        audio, files = _visible_result(record)
        state = _progress_state(record)
        yield audio, files, f"模型或音色载入失败：{html.escape(str(exc))}", _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=False), _retry_button_update(state)
        return

    params = record["params"]
    kwargs: dict[str, Any] = {
        "do_sample": bool(params["do_sample"]),
        "repetition_penalty": float(params["repetition_penalty"]),
        "max_new_tokens": int(params["max_new_tokens"]),
    }
    if params["do_sample"]:
        kwargs.update(temperature=float(params["temperature"]), top_p=float(params["top_p"]), top_k=int(params["top_k"]))

    total = len(indexes)
    stopped = False
    for position, line_index in enumerate(indexes, start=1):
        if STOP_REQUESTED.is_set():
            stopped = True
            break
        line = next((item for item in record["lines"] if item["index"] == line_index), None)
        if not line or line.get("status") == "已完成":
            continue
        line["status"] = "进行中"
        line["error"] = ""
        _write_history(record)
        progress((position - 1) / max(total, 1), desc=f"正在合成 {position}/{total}")
        state = _progress_state(record)
        yield *_visible_result(record), f"正在合成第 {line_index}/{len(record['lines'])} 段…", _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=True), _retry_button_update(state, running=True)
        try:
            with INFERENCE_LOCK:
                _ensure_model(voice_mode)
                if voice_mode == "clone":
                    waveforms, sample_rate = MODEL.generate_voice_clone(
                        text=line["text"], language=params.get("language") or "Auto",
                        voice_clone_prompt=prompt_items, **kwargs,
                    )
                else:
                    waveforms, sample_rate = MODEL.generate_custom_voice(
                        text=line["text"], language=params.get("language") or "Auto",
                        speaker=record["speaker"], instruct=params.get("instruction") or None,
                        **kwargs,
                    )
            output_name = f"{batch_id}_{line_index:02d}.wav"
            output_path = OUTPUT_DIR / output_name
            sf.write(str(output_path), waveforms[0], sample_rate, subtype="PCM_24")
            line["status"] = "已完成"
            line["output_file"] = output_name
            line["error"] = ""
            LOG.info("Generated batch %s line %s", batch_id, line_index)
        except Exception as exc:
            LOG.exception("Speech generation failed in batch %s line %s", batch_id, line_index)
            line["status"] = "失败"
            line["error"] = str(exc)[:500]
        _write_history(record)
        progress(position / max(total, 1), desc=f"已处理 {position}/{total}")
        audio, files = _visible_result(record)
        state = _progress_state(record)
        yield audio, files, _batch_message(record), _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=True), _retry_button_update(state, running=True)

    if STOP_REQUESTED.is_set():
        stopped = True
    _write_history(record)
    audio, files = _visible_result(record)
    status = _batch_message(record, stopped=stopped)
    state = _progress_state(record)
    yield audio, files, status, _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=False), _retry_button_update(state)


def _error_generation(message: str):
    yield None, [], message, [], {}, gr.update(choices=history_choices()), gr.update(visible=False), gr.update(visible=False)


def synthesize(
    voice_mode: str, profile_id: str, custom_voice_speaker: str, custom_voice_instruction: str,
    script: str, language: str, do_sample: bool, temperature: float,
    top_p: float, top_k: int, repetition_penalty: float, max_new_tokens: int,
    progress=gr.Progress(),
):
    lines = [line.strip() for line in (script or "").splitlines() if line.strip()]
    if not lines:
        yield from _error_generation("请先输入要合成的文本。")
        return
    if len(lines) > 8:
        yield from _error_generation("每批最多 8 行，请分批生成。")
        return
    if any(len(line) > 4000 for line in lines):
        yield from _error_generation("单行文本超过 4,000 字符，请拆成更短的语段。")
        return
    if voice_mode not in {"clone", "custom_voice"}:
        yield from _error_generation("请选择一种音色模式。")
        return
    if voice_mode == "clone":
        if not profile_id:
            yield from _error_generation("请先选择或添加一个音色。")
            return
        metadata = next((item for item in list_profiles() if item["id"] == profile_id), None)
        if not metadata:
            yield from _error_generation("所选音色不存在，请刷新音色列表。")
            return
        record_voice = {
            "profile_id": profile_id,
            "profile_name": metadata["name"],
            "profile_mode": metadata.get("mode"),
        }
        voice_params: dict[str, Any] = {}
    else:
        if custom_voice_speaker not in CUSTOM_VOICE_LABELS:
            yield from _error_generation("请选择一个官方预制音色。")
            return
        record_voice = {
            "profile_id": None,
            "profile_name": CUSTOM_VOICE_LABELS[custom_voice_speaker],
            "profile_mode": None,
            "speaker": custom_voice_speaker,
        }
        voice_params = {"instruction": (custom_voice_instruction or "").strip()}
    record = {
        "schema_version": 1, "id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_mode": voice_mode, **record_voice,
        "params": {
            "language": language or "Auto", "do_sample": bool(do_sample),
            "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty), "max_new_tokens": int(max_new_tokens),
            **voice_params,
        },
        "lines": [{"index": index, "text": text, "status": "待生成", "output_file": None, "error": ""}
                  for index, text in enumerate(lines, start=1)],
    }
    _write_history(record)
    yield from _stream_batch(record, [line["index"] for line in record["lines"]], progress)


def retry_incomplete(retry_state: dict[str, Any] | None, progress=gr.Progress()):
    try:
        if not retry_state or not retry_state.get("batch_id"):
            yield from _error_generation("没有待重试或继续的语段。")
            return
        record = _read_history(retry_state["batch_id"])
        indexes = [int(value) for value in retry_state.get("indexes", [])]
        allowed = {item["index"] for item in record["lines"] if item.get("status") != "已完成"}
        indexes = [value for value in indexes if value in allowed]
        if not indexes:
            indexes = sorted(allowed)
        if not indexes:
            yield from _error_generation("这批语段已全部完成。")
            return
        yield from _stream_batch(record, indexes, progress)
    except Exception as exc:
        LOG.exception("Could not resume generation history")
        yield from _error_generation(f"无法继续这条记录：{html.escape(str(exc))}")


def request_stop():
    STOP_REQUESTED.set()
    return "将在当前语段结束后停止，已完成的音频会保留。"


def _outputs_for_generation():
    return [result_audio, output_files, generation_status, line_table, retry_state, history_picker, stop_button, retry_button]


initial_profiles = profile_choices()
initial_profile_id = initial_profiles[0][1] if initial_profiles else None
recover_interrupted_history()

with gr.Blocks(title="Qwen3-TTS") as app:
    gr.HTML(HEADER_HTML)
    with gr.Tabs():
        with gr.Tab("工作台", id="workbench"):
            with gr.Row(elem_classes=["studio-grid"]):
                with gr.Column(scale=4, min_width=310, elem_classes=["panel"]):
                    voice_mode = gr.Radio(
                        choices=[("音色克隆", "clone"), ("官方预制音色", "custom_voice")],
                        value="clone", label="音色模式", elem_id="voice-mode",
                    )
                    with gr.Group(visible=True) as clone_voice_panel:
                        gr.HTML('<div class="section-title">参考音频克隆</div><div class="section-note">选择或管理可复用的音色档案。</div>')
                        profile_guide = gr.HTML(
                            '<div class="warm-note"><strong>添加第一个音色</strong><br>上传或录制一段清晰的人声，填写逐字稿后保存。</div>',
                            elem_id="first-use-guide", visible=not initial_profiles,
                        )
                        voice_radio = gr.Radio(
                            choices=initial_profiles, value=initial_profile_id, label="选择音色",
                            elem_id="voice-cards", container=False,
                        )
                        profile_status = gr.Markdown("", elem_classes=["status-box"])
                        profile_preview = gr.Audio(
                            label="参考音频试听", type="filepath", interactive=False,
                            autoplay=False, elem_classes=["audio-panel"],
                        )
                        with gr.Accordion("管理当前音色", open=False):
                            rename_name = gr.Textbox(label="音色名称", max_lines=1)
                            with gr.Row():
                                rename_button = gr.Button("保存名称", size="sm", variant="secondary")
                                export_button = gr.Button("导出档案", size="sm", variant="secondary")
                                trash_button = gr.Button("移入回收站", size="sm", variant="secondary")
                            export_status = gr.Markdown("", elem_classes=["section-note"])
                            export_file = gr.File(label="档案 ZIP", interactive=False, visible=False)
                        with gr.Accordion("回收站", open=False):
                            trash_dropdown = gr.Dropdown(choices=trash_choices(), label="已移入回收站的音色", value=None)
                            restore_button = gr.Button("恢复所选音色", size="sm", variant="secondary")
                        refresh_button = gr.Button("刷新音色列表", size="sm", variant="secondary")
                        with gr.Accordion("添加音色", open=not initial_profiles):
                            gr.Markdown("选择一段清晰的单人语音。ICL 需要逐字稿；仅提取音色不需要。", elem_classes=["section-note"])
                            reference_audio = gr.Audio(
                                sources=["upload", "microphone"], type="filepath", label="参考音频",
                                elem_classes=["audio-panel"],
                            )
                            clone_mode = gr.Radio(choices=CLONE_MODES, value="icl", label="克隆方式")
                            reference_text = gr.Textbox(
                                label="参考音频逐字稿", placeholder="准确写下录音里说的内容。",
                                lines=3, visible=True,
                            )
                            new_profile_name = gr.Textbox(label="新音色名称", placeholder="例如：中文旁白", max_lines=1)
                            save_profile_button = gr.Button("保存音色", variant="primary")
                    with gr.Group(visible=False) as custom_voice_panel:
                        gr.HTML('<div class="section-title">官方预制音色</div><div class="section-note">选择内置说话人，并可通过风格指令调整表达方式。</div>')
                        custom_voice_speaker = gr.Dropdown(
                            choices=CUSTOM_VOICE_CHOICES, value="Uncle_Fu", label="官方音色",
                            filterable=False,
                        )
                        custom_voice_instruction = gr.Textbox(
                            label="风格指令", value=DEFAULT_CUSTOM_VOICE_INSTRUCTION,
                            placeholder="例如：语速舒缓，情绪克制，像纪录片旁白。", lines=4,
                        )

                with gr.Column(scale=7, min_width=430, elem_classes=["panel"]):
                    gr.HTML('<div class="section-title">语音合成</div><div class="section-note">逐行生成 · 每批最多 8 段。</div>')
                    script_box = gr.Textbox(
                        label="合成文本", placeholder="输入一行或多行文字……\n每行会单独合成为一段 WAV。",
                        lines=9, max_lines=20, value="", elem_id="script-box",
                    )
                    script_stats = gr.Markdown("输入文本后，每一行会生成一段语音。", elem_classes=["section-note"])
                    with gr.Row(elem_id="speech-options"):
                        language_dropdown = gr.Dropdown(choices=LANGUAGES, value="Auto", label="语音语言", scale=3, filterable=False)
                        do_sample = gr.Checkbox(value=True, label="随机采样", scale=2)
                    sampling_preset = gr.Dropdown(
                        choices=[("默认", "default"), ("较低随机度", "lower_randomness"), ("较高随机度", "higher_randomness")],
                        value="default", label="快速预设", filterable=False,
                    )
                    with gr.Accordion("采样参数", open=False):
                        with gr.Row():
                            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.9, step=0.05, label="Temperature · 变化度")
                            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p · 采样范围")
                        with gr.Row():
                            top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k · 候选数量")
                            repetition_penalty = gr.Slider(minimum=1.0, maximum=1.5, value=1.05, step=0.01, label="重复抑制")
                        max_new_tokens = gr.Slider(minimum=128, maximum=4096, value=2048, step=128, label="最大生成长度")
                        gr.Markdown("预设只调整采样项，可继续微调。关闭随机采样后，温度、Top-p 和 Top-k 不参与生成。", elem_classes=["section-note"])
                    with gr.Row():
                        generate_button = gr.Button("开始生成", variant="primary", elem_id="generate-button", scale=4)
                        stop_button = gr.Button("停止", variant="secondary", elem_id="stop-button", scale=1, visible=False)
                    generation_status = gr.Markdown("等待输入文本。", elem_classes=["status-box"])
                    line_table = gr.Dataframe(
                        headers=["段落", "文本", "状态", "文件", "错误"], value=[], interactive=False,
                        wrap=True, label="逐段进度", elem_id="line-table",
                    )
                    retry_state = gr.State({})
                    retry_button = gr.Button("重试 / 继续未完成", variant="secondary", visible=False)
                    gr.HTML('<div class="section-title output-title">生成结果</div>')
                    result_audio = gr.Audio(label="试听第一段", type="filepath", interactive=False, autoplay=False, elem_classes=["audio-panel"])
                    output_files = gr.Files(label="本批 WAV 文件", file_count="multiple", type="filepath", interactive=False)

        with gr.Tab("生成记录", id="history"):
            with gr.Column(elem_classes=["panel"]):
                gr.HTML('<div class="section-title">生成记录</div><div class="section-note">文本、设置和逐段结果保存在外接盘，可试听、下载或载入参数。</div>')
                with gr.Row():
                    history_picker = gr.Dropdown(choices=history_choices(), label="选择一批生成记录", scale=5)
                    history_refresh = gr.Button("刷新", size="sm", variant="secondary", scale=1)
                history_status = gr.Markdown("选择一条记录查看结果。", elem_classes=["status-box"])
                history_audio = gr.Audio(label="试听", type="filepath", interactive=False, autoplay=False, elem_classes=["audio-panel"])
                history_files = gr.Files(label="本批音频", file_count="multiple", type="filepath", interactive=False)
                history_table = gr.Dataframe(
                    headers=["段落", "文本", "状态", "文件", "错误"], value=[], interactive=False,
                    wrap=True, label="逐段结果", elem_id="history-table",
                )
                load_settings_button = gr.Button("载入这批设置到工作台", variant="secondary")
                history_load_status = gr.Markdown("", elem_classes=["section-note"])

    voice_mode.change(
        fn=lambda value: (gr.update(visible=(value == "clone")), gr.update(visible=(value == "custom_voice"))),
        inputs=[voice_mode], outputs=[clone_voice_panel, custom_voice_panel], queue=False,
    )
    clone_mode.change(lambda value: gr.update(visible=(value == "icl")), inputs=[clone_mode], outputs=[reference_text], queue=False)
    sampling_preset.change(
        fn=apply_sampling_preset, inputs=[sampling_preset],
        outputs=[do_sample, temperature, top_p, top_k, repetition_penalty, max_new_tokens], queue=False,
    )
    voice_radio.change(fn=load_profile_status, inputs=[voice_radio], outputs=[profile_status, profile_preview, rename_name])
    refresh_button.click(fn=refresh_profiles, inputs=[voice_radio], outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown], queue=False)
    save_profile_button.click(
        fn=save_voice_profile, inputs=[new_profile_name, reference_audio, reference_text, clone_mode],
        outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown],
    )
    rename_button.click(
        fn=rename_voice_profile, inputs=[voice_radio, rename_name],
        outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown], queue=False,
    )
    export_button.click(fn=export_voice_profile, inputs=[voice_radio], outputs=[export_file, export_status], queue=False)
    trash_button.click(
        fn=move_profile_to_trash, inputs=[voice_radio],
        outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown], queue=False,
    )
    restore_button.click(
        fn=restore_profile, inputs=[trash_dropdown],
        outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown], queue=False,
    )
    script_box.input(
        fn=update_script_stats, inputs=[script_box], outputs=[script_stats, generate_button],
        queue=False, trigger_mode="always_last",
    )
    script_box.change(
        fn=update_script_stats, inputs=[script_box], outputs=[script_stats, generate_button],
        queue=False, trigger_mode="always_last",
    )
    generate_button.click(
        fn=synthesize,
        inputs=[voice_mode, voice_radio, custom_voice_speaker, custom_voice_instruction,
                script_box, language_dropdown, do_sample, temperature, top_p, top_k, repetition_penalty, max_new_tokens],
        outputs=_outputs_for_generation(),
    )
    retry_button.click(fn=retry_incomplete, inputs=[retry_state], outputs=_outputs_for_generation())
    stop_button.click(fn=request_stop, inputs=[], outputs=[generation_status], queue=False)
    history_picker.change(fn=load_history_view, inputs=[history_picker], outputs=[history_status, history_audio, history_files, history_table])
    history_refresh.click(fn=lambda: gr.update(choices=history_choices()), inputs=[], outputs=[history_picker], queue=False)
    load_settings_button.click(
        fn=load_history_settings, inputs=[history_picker],
        outputs=[voice_mode, clone_voice_panel, custom_voice_panel, voice_radio, custom_voice_speaker,
                 custom_voice_instruction, script_box, language_dropdown, do_sample, temperature, top_p,
                 top_k, repetition_penalty, max_new_tokens, history_load_status],
    )
    app.load(
        fn=initialize_page,
        inputs=[voice_radio],
        outputs=[voice_radio, profile_status, profile_preview, rename_name, profile_guide, trash_dropdown, history_picker],
    )

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=32)
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("QWEN_TTS_PORT", "8000")),
        share=False,
        inbrowser=False,
        quiet=True,
        show_error=False,
        max_file_size="100mb",
        allowed_paths=[str(VOICE_DIR), str(OUTPUT_DIR), str(EXPORT_DIR)],
        css=CSS,
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="stone"),
    )
