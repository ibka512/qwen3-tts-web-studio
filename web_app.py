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
os.environ.setdefault("TMPDIR", str(ROOT / "cache" / "tmp"))
MODEL_DIR = Path(
    os.environ.get("QWEN_TTS_MODEL_DIR", ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
).expanduser().resolve()
CUSTOMVOICE_MODEL_DIR = Path(
    os.environ.get("QWEN_TTS_CUSTOM_MODEL_DIR", ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice")
).expanduser().resolve()
VOICE_DESIGN_MODEL_DIR = Path(
    os.environ.get("QWEN_TTS_VOICEDESIGN_MODEL_DIR", ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
).expanduser().resolve()
VOICE_DIR = ROOT / "voices"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
HISTORY_DIR = ROOT / "history"
EXPORT_DIR = ROOT / "exports"
TRASH_DIR = VOICE_DIR / ".trash"
for directory in (VOICE_DIR, OUTPUT_DIR, LOG_DIR, HISTORY_DIR, EXPORT_DIR, TRASH_DIR, ROOT / "cache" / "gradio", ROOT / "cache" / "tmp"):
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
    ("录音与原文", "icl"),
    ("仅参考声音", "x_vector"),
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
    if kind not in {"clone", "custom_voice", "voice_design"}:
        raise ValueError("不支持的音色模式。")
    if MODEL is not None and MODEL_KIND == kind:
        return MODEL

    model_dir = {"clone": MODEL_DIR, "custom_voice": CUSTOMVOICE_MODEL_DIR, "voice_design": VOICE_DESIGN_MODEL_DIR}[kind]
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

    label = {"clone": "Base", "custom_voice": "CustomVoice", "voice_design": "VoiceDesign"}[kind]
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
    found = [metadata for folder in VOICE_DIR.iterdir() if folder.is_dir() and folder.name != ".trash" and (folder / "profile.json").is_file()
             if (metadata := _read_profile(folder))]
    return sorted(found, key=lambda item: item.get("created_at", ""), reverse=True)


def list_trashed_profiles() -> list[dict[str, Any]]:
    found = [metadata for folder in TRASH_DIR.iterdir() if folder.is_dir() and (folder / "profile.json").is_file()
             if (metadata := _read_profile(folder))]
    return sorted(found, key=lambda item: item.get("trashed_at", ""), reverse=True)


def profile_choices() -> list[tuple[str, str]]:
    return [(item["name"], item["id"]) for item in list_profiles()]


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
        return "", None, ""
    try:
        load_prompt(profile_id)
        metadata = next((item for item in list_profiles() if item["id"] == profile_id), None)
        name = metadata["name"] if metadata else "已选音色"
        mode = "录音与原文" if metadata and metadata.get("mode") == "icl" else "仅参考声音"
        return f"**{html.escape(name)} 已就绪** · {mode}", _profile_audio_path(profile_id), name
    except Exception:
        LOG.exception("Failed to load voice profile %s", profile_id)
        return "这条音色暂时无法使用，请刷新列表或重新选择。", _profile_audio_path(profile_id), ""


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
        return (gr.update(choices=profile_choices()), "选择“录音与原文”时，请填写录音内容。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))

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
    except Exception:
        LOG.exception("Could not create voice profile %s", clean_name)
        PROMPT_CACHE.pop(profile_id, None)
        shutil.rmtree(profile_dir, ignore_errors=True)
        return (gr.update(choices=profile_choices()), "音色保存失败，请检查参考音频后重试。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))


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
    except Exception:
        LOG.exception("Could not rename voice profile %s", profile_id)
        return (gr.update(choices=profile_choices()), "改名失败，请重试。", _profile_audio_path(profile_id), clean_name, gr.update(visible=False), gr.update(choices=trash_choices()))


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
    except Exception:
        LOG.exception("Could not move voice profile %s to trash", profile_id)
        return (gr.update(choices=choices), "无法移入回收站，请重试。", _profile_audio_path(profile_id), "", gr.update(visible=not choices), gr.update(choices=trash_choices()))


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
    except Exception:
        LOG.exception("Could not restore voice profile %s", profile_id)
        return (gr.update(choices=profile_choices()), "恢复失败，请重试。", None, "", gr.update(visible=not list_profiles()), gr.update(choices=trash_choices()))


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
    except Exception:
        LOG.exception("Could not export voice profile %s", profile_id)
        return gr.update(value=None, visible=False), "导出失败，请重试。"


def update_script_stats(script: str):
    lines = [line.strip() for line in (script or "").splitlines() if line.strip()]
    too_long = sum(len(line) > 4000 for line in lines)
    if not lines:
        return "0 / 8 段", gr.update(interactive=False)
    if len(lines) > 8:
        return f"{len(lines)} / 8 段 · 请减少或合并段落", gr.update(interactive=False)
    if too_long:
        return f"有 {too_long} 段超过 4,000 字，请拆分后再生成", gr.update(interactive=False)
    count = sum(len(line) for line in lines)
    return f"{len(lines)} / 8 段 · {count:,} 字", gr.update(interactive=True)


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
        if path.name.startswith("._"):
            continue
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
        if path.name.startswith("._"):
            continue
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
             str(item.get("output_file") or ""),
             ("生成失败，可载入设置后重新生成。" if item.get("status") == "失败" and item.get("error") else "")]
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
    except Exception:
        LOG.exception("Could not load generation history %s", batch_id)
        return "读取记录失败，请刷新记录列表后重试。", None, [], []


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
    except Exception:
        LOG.exception("Could not load generation settings %s", batch_id)
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), DEFAULT_CUSTOM_VOICE_INSTRUCTION,
            "", gr.update(), True, 0.9, 1.0, 50, 1.05, 2048,
            "无法载入设置，请刷新记录后重试。",
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
        return f"已暂停 · 完成 {completed}/{len(lines)} 段。已完成的音频已保存。"
    if failed or pending:
        return f"完成 {completed}/{len(lines)} 段 · {failed} 段失败 · {pending} 段待生成。"
    return f"已完成 {completed} 段 · 可试听或下载。"


def _stream_batch(record: dict[str, Any], indexes: list[int], progress):
    STOP_REQUESTED.clear()
    batch_id = record["id"]
    rows = _history_rows(record)
    audio, files = _visible_result(record)
    state = _progress_state(record)
    voice_mode = record.get("model_mode", "clone")
    if voice_mode not in {"clone", "custom_voice"}:
        voice_mode = "clone"
    yield audio, files, "正在加载声音…", rows, state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=True), _retry_button_update(state, running=True)
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
        yield audio, files, "声音载入失败，请检查音色后重试。", _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=False), _retry_button_update(state)
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
        yield *_visible_result(record), f"正在生成第 {line_index}/{len(record['lines'])} 段…", _history_rows(record), state, gr.update(choices=history_choices(), value=batch_id), gr.update(visible=True), _retry_button_update(state, running=True)
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
        yield from _error_generation("请选择声音来源。")
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
            yield from _error_generation("请选择一个内置音色。")
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
    except Exception:
        LOG.exception("Could not resume generation history")
        yield from _error_generation("无法继续这批口播，请刷新记录后重试。")


def request_stop():
    STOP_REQUESTED.set()
    return "正在完成当前段，随后暂停。已完成的音频会保留。"


def _outputs_for_generation():
    return [result_audio, output_files, generation_status, line_table, retry_state, history_picker, stop_button, retry_button]


from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


TASK_LOCK = threading.Lock()
ACTIVE_TASK_ID: str | None = None
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
UPLOAD_DIR = ROOT / "cache" / "uploads"
VOICE_DESIGN_PREVIEW_DIR = ROOT / "cache" / "voice-design-previews"
for directory in (UPLOAD_DIR, VOICE_DESIGN_PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _active_task() -> str | None:
    with TASK_LOCK:
        return ACTIVE_TASK_ID


def _claim_task(task_id: str) -> None:
    global ACTIVE_TASK_ID
    with TASK_LOCK:
        if ACTIVE_TASK_ID:
            raise HTTPException(status_code=409, detail="当前有任务正在生成，请等它完成或暂停后再试。")
        ACTIVE_TASK_ID = task_id


def _release_task(task_id: str) -> None:
    global ACTIVE_TASK_ID
    with TASK_LOCK:
        if ACTIVE_TASK_ID == task_id:
            ACTIVE_TASK_ID = None


def _profile_payload(metadata: dict[str, Any], trashed: bool = False) -> dict[str, Any]:
    return {
        "id": metadata["id"],
        "name": metadata["name"],
        "mode": metadata.get("mode", "icl"),
        "ref_text": metadata.get("ref_text", ""),
        "created_at": metadata.get("created_at", ""),
        "voice_design": metadata.get("voice_design"),
        "trashed": trashed,
        "audio_url": f"/api/voices/{metadata['id']}/audio" + ("?trash=1" if trashed else ""),
    }


def _profile_audio_file(profile_id: str, trashed: bool = False) -> Path:
    folder = _profile_dir(profile_id, allow_trash=trashed)
    metadata = _read_profile(folder)
    if not metadata:
        raise HTTPException(status_code=404, detail="音色档案或参考音频不存在。")
    return folder / metadata["audio_file"]


def _history_payload(record: dict[str, Any]) -> dict[str, Any]:
    public = dict(record)
    lines: list[dict[str, Any]] = []
    for source in record.get("lines", []):
        line = dict(source)
        safe_file = _safe_output_path(line.get("output_file"))
        line["audio_url"] = f"/api/files/{Path(safe_file).name}" if safe_file else None
        lines.append(line)
    public["lines"] = lines
    public["progress"] = (
        sum(line.get("status") == "已完成" for line in lines) / max(len(lines), 1)
    )
    active = _active_task() == record.get("id")
    if active:
        public["state"] = "running"
    elif lines and all(line.get("status") == "已完成" for line in lines):
        public["state"] = "completed"
    elif any(line.get("status") == "失败" for line in lines):
        public["state"] = "partial" if any(line.get("status") == "已完成" for line in lines) else "failed"
    elif any(line.get("status") in {"待生成", "进行中"} for line in lines):
        public["state"] = "paused"
    else:
        public["state"] = "completed"
    return public


def _sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        temperature = float(payload.get("temperature", 0.9))
        top_p = float(payload.get("top_p", 1.0))
        top_k = int(payload.get("top_k", 50))
        repetition_penalty = float(payload.get("repetition_penalty", 1.05))
        max_new_tokens = int(payload.get("max_new_tokens", 2048))
        subtalker_temperature = float(payload.get("subtalker_temperature", 0.9))
        subtalker_top_p = float(payload.get("subtalker_top_p", 1.0))
        subtalker_top_k = int(payload.get("subtalker_top_k", 50))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="高级参数格式无效。")
    if not 0.1 <= temperature <= 1.5:
        raise HTTPException(status_code=422, detail="变化幅度需在 0.1 至 1.5 之间。")
    if not 0.1 <= top_p <= 1.0:
        raise HTTPException(status_code=422, detail="采样范围需在 0.1 至 1.0 之间。")
    if not 1 <= top_k <= 100:
        raise HTTPException(status_code=422, detail="候选数量需在 1 至 100 之间。")
    if not 1.0 <= repetition_penalty <= 1.5:
        raise HTTPException(status_code=422, detail="重复抑制需在 1.0 至 1.5 之间。")
    if not 128 <= max_new_tokens <= 4096:
        raise HTTPException(status_code=422, detail="单段长度上限需在 128 至 4,096 之间。")
    if not 0.1 <= subtalker_temperature <= 1.5 or not 0.1 <= subtalker_top_p <= 1.0 or not 1 <= subtalker_top_k <= 100:
        raise HTTPException(status_code=422, detail="声音细节参数超出范围。")
    return {
        "do_sample": bool(payload.get("do_sample", True)),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "subtalker_dosample": bool(payload.get("subtalker_dosample", True)),
        "subtalker_temperature": subtalker_temperature,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_top_k": subtalker_top_k,
    }


def _line_voice(line: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    saved = line.get("voice")
    if isinstance(saved, dict):
        mode = saved.get("mode", record.get("model_mode", "clone"))
        profile_id = saved.get("profile_id", record.get("profile_id"))
        speaker = saved.get("speaker", record.get("speaker"))
        instruction = saved.get("instruction", record.get("params", {}).get("instruction", ""))
        language = saved.get("language", record.get("params", {}).get("language", "Auto"))
        return {
            "mode": mode,
            "profile_id": profile_id,
            "speaker": speaker,
            "instruction": instruction,
            "language": language,
        }
    return {
        "mode": record.get("model_mode", "clone"),
        "profile_id": record.get("profile_id"),
        "speaker": record.get("speaker"),
        "instruction": record.get("params", {}).get("instruction", ""),
        "language": record.get("params", {}).get("language", "Auto"),
    }


def _run_api_job(batch_id: str, indexes: list[int]) -> None:
    try:
        record = _read_history(batch_id)
        for line_index in indexes:
            if STOP_REQUESTED.is_set():
                break
            line = next((item for item in record["lines"] if item.get("index") == line_index), None)
            if not line or line.get("status") == "已完成":
                continue
            voice = _line_voice(line, record)
            mode = voice["mode"]
            line["status"] = "进行中"
            line["error"] = ""
            _write_history(record)
            try:
                if mode not in {"clone", "custom_voice"}:
                    raise ValueError("这段的声音来源无效。")
                with INFERENCE_LOCK:
                    _ensure_model(mode)
                    if mode == "clone":
                        prompt_items = load_prompt(voice["profile_id"], allow_trash=True)
                        waves, sample_rate = MODEL.generate_voice_clone(
                            text=line["text"],
                            language=voice["language"] or "Auto",
                            voice_clone_prompt=prompt_items,
                            **{key: value for key, value in record["params"].items() if key not in {"language", "instruction"}},
                        )
                    else:
                        speaker = voice["speaker"]
                        if speaker not in CUSTOM_VOICE_LABELS:
                            raise ValueError("请选择有效的官方预制音色。")
                        waves, sample_rate = MODEL.generate_custom_voice(
                            text=line["text"],
                            language=voice["language"] or "Auto",
                            speaker=speaker,
                            instruct=(voice.get("instruction") or None),
                            **{key: value for key, value in record["params"].items() if key not in {"language", "instruction"}},
                        )
                output_name = f"{batch_id}_{int(line_index):02d}.wav"
                sf.write(str(OUTPUT_DIR / output_name), waves[0], sample_rate, subtype="PCM_24")
                line["status"] = "已完成"
                line["output_file"] = output_name
                line["error"] = ""
                LOG.info("Generated batch %s line %s", batch_id, line_index)
            except Exception as exc:
                LOG.exception("Speech generation failed in batch %s line %s", batch_id, line_index)
                line["status"] = "失败"
                line["error"] = str(exc)[:500]
            _write_history(record)
    except Exception:
        LOG.exception("Generation job failed for batch %s", batch_id)
        try:
            record = _read_history(batch_id)
            for line in record.get("lines", []):
                if line.get("status") == "进行中":
                    line["status"] = "失败"
                    line["error"] = "任务中断，可重试此段。"
            _write_history(record)
        except Exception:
            LOG.exception("Could not save failed generation state for %s", batch_id)
    finally:
        _release_task(batch_id)


def _launch_api_job(record: dict[str, Any], indexes: list[int]) -> None:
    batch_id = record["id"]
    _claim_task(batch_id)
    STOP_REQUESTED.clear()
    thread = threading.Thread(target=_run_api_job, args=(batch_id, indexes), daemon=True)
    try:
        thread.start()
    except Exception:
        _release_task(batch_id)
        raise


recover_interrupted_history()

app = FastAPI(title="Qwen3-TTS Local Studio", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def studio_home():
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


@app.get("/api/state")
def studio_state():
    active_task = _active_task()
    active_generation_id = active_task if active_task and re.fullmatch(r"[a-f0-9]{32}", active_task) and _manifest_path(active_task).is_file() else None
    return {
        "active_task": active_task,
        "active_generation_id": active_generation_id,
        "busy": active_task is not None,
        "model_kind": MODEL_KIND,
        "models": {
            "clone": (MODEL_DIR / "model.safetensors").is_file(),
            "custom_voice": (CUSTOMVOICE_MODEL_DIR / "model.safetensors").is_file(),
            "voice_design": (VOICE_DESIGN_MODEL_DIR / "model.safetensors").is_file(),
        },
    }


@app.get("/api/bootstrap")
def bootstrap():
    return {
        "languages": [{"label": label, "value": value} for label, value in LANGUAGES],
        "speakers": [
            {"label": label, "value": value, "gender": "女声" if value in {"Vivian", "Serena", "Ono_Anna", "Sohee"} else "男声"}
            for label, value in CUSTOM_VOICE_CHOICES
        ],
        "models": {
            "clone": (MODEL_DIR / "model.safetensors").is_file(),
            "custom_voice": (CUSTOMVOICE_MODEL_DIR / "model.safetensors").is_file(),
            "voice_design": (VOICE_DESIGN_MODEL_DIR / "model.safetensors").is_file(),
        },
        "defaults": {
            "instruction": DEFAULT_CUSTOM_VOICE_INSTRUCTION,
            "temperature": 0.9,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "max_new_tokens": 2048,
        },
    }


@app.get("/api/voices")
def get_voices():
    return {
        "profiles": [_profile_payload(item) for item in list_profiles()],
        "trash": [_profile_payload(item, trashed=True) for item in list_trashed_profiles()],
    }


@app.get("/api/voices/{profile_id}/audio")
def get_voice_audio(profile_id: str, trash: bool = False):
    path = _profile_audio_file(profile_id, trashed=trash)
    return FileResponse(path, media_type="audio/wav" if path.suffix.lower() == ".wav" else None)


@app.post("/api/voices")
def create_voice(
    name: str = Form(...),
    mode: str = Form("icl"),
    ref_text: str = Form(""),
    audio: UploadFile = File(...),
):
    if mode not in {"icl", "x_vector"}:
        raise HTTPException(status_code=422, detail="请选择参考音频方式。")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ACCEPTED_AUDIO:
        raise HTTPException(status_code=415, detail="音频格式需为 WAV、FLAC、MP3、M4A、OGG、AIFF 或 WEBM。")
    task_id = "voice-save:" + uuid.uuid4().hex
    _claim_task(task_id)
    source = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        content = audio.file.read(MAX_UPLOAD_BYTES + 1)
        audio.file.close()
        if not content:
            raise HTTPException(status_code=400, detail="参考音频为空。")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="参考音频超过 100 MB。")
        before = {item["id"] for item in list_profiles()}
        source.write_bytes(content)
        result = save_voice_profile(name, str(source), ref_text, mode, progress=lambda *args, **kwargs: None)
        created = [item for item in list_profiles() if item["id"] not in before]
        if not created:
            raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
        return _profile_payload(created[0])
    finally:
        source.unlink(missing_ok=True)
        _release_task(task_id)

@app.patch("/api/voices/{profile_id}")
def rename_voice(profile_id: str, payload: dict[str, Any]):
    result = rename_voice_profile(profile_id, str(payload.get("name", "")))
    if "已改名为" not in str(result[1]):
        raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
    metadata = next((item for item in list_profiles() if item["id"] == profile_id), None)
    if not metadata:
        raise HTTPException(status_code=404, detail="音色档案不存在。")
    return _profile_payload(metadata)


@app.post("/api/voices/{profile_id}/trash")
def trash_voice(profile_id: str):
    result = move_profile_to_trash(profile_id)
    if "已移入回收站" not in str(result[1]):
        raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
    metadata = next((item for item in list_trashed_profiles() if item["id"] == profile_id), None)
    return _profile_payload(metadata, trashed=True) if metadata else {"id": profile_id, "trashed": True}


@app.post("/api/voices/{profile_id}/restore")
def restore_voice(profile_id: str):
    result = restore_profile(profile_id)
    if "已恢复" not in str(result[1]):
        raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
    metadata = next((item for item in list_profiles() if item["id"] == profile_id), None)
    return _profile_payload(metadata) if metadata else {"id": profile_id, "trashed": False}


@app.post("/api/voices/{profile_id}/export")
def export_voice(profile_id: str):
    result = export_voice_profile(profile_id)
    value = result[0].get("value") if isinstance(result[0], dict) else None
    path = Path(value) if value else None
    if not path or not path.is_file() or path.resolve().parent != EXPORT_DIR.resolve():
        raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
    return {"filename": path.name, "download_url": f"/api/exports/{path.name}"}


@app.get("/api/exports/{filename}")
def get_export(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="文件不存在。")
    path = EXPORT_DIR / filename
    if path.resolve().parent != EXPORT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(path, filename=path.name, media_type="application/zip")


@app.post("/api/voices/design/preview")
def create_voice_design_preview(payload: dict[str, Any]):
    if not (VOICE_DESIGN_MODEL_DIR / "model.safetensors").is_file():
        raise HTTPException(status_code=409, detail="VoiceDesign 权重尚未下载完成。")
    text = str(payload.get("text", "")).strip()
    instruction = str(payload.get("instruction", "")).strip()
    language = str(payload.get("language", "Chinese"))
    if not text or len(text) > 1000:
        raise HTTPException(status_code=422, detail="试听文案需为 1 至 1,000 字符。")
    if not instruction or len(instruction) > 1200:
        raise HTTPException(status_code=422, detail="音色描述需为 1 至 1,200 字符。")
    if language not in {value for _, value in LANGUAGES} - {"Auto"}:
        raise HTTPException(status_code=422, detail="请选择支持的语言。")
    params = _sampling_params(payload)
    preview_id = uuid.uuid4().hex
    task_id = f"design:{preview_id}"
    _claim_task(task_id)
    try:
        with INFERENCE_LOCK:
            _ensure_model("voice_design")
            waves, sample_rate = MODEL.generate_voice_design(
                text=text,
                instruct=instruction,
                language=language,
                **params,
            )
        audio_path = VOICE_DESIGN_PREVIEW_DIR / f"{preview_id}.wav"
        sf.write(str(audio_path), waves[0], sample_rate, subtype="PCM_24")
        manifest = {
            "id": preview_id,
            "text": text,
            "instruction": instruction,
            "language": language,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (VOICE_DESIGN_PREVIEW_DIR / f"{preview_id}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {**manifest, "audio_url": f"/api/voice-design/{preview_id}/audio"}
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("VoiceDesign preview failed")
        raise HTTPException(status_code=500, detail=f"试听生成失败：{str(exc)[:240]}")
    finally:
        _release_task(task_id)


@app.get("/api/voice-design/{preview_id}/audio")
def get_voice_design_audio(preview_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", preview_id):
        raise HTTPException(status_code=404, detail="试听不存在。")
    path = VOICE_DESIGN_PREVIEW_DIR / f"{preview_id}.wav"
    if not path.is_file() or path.resolve().parent != VOICE_DESIGN_PREVIEW_DIR.resolve():
        raise HTTPException(status_code=404, detail="试听不存在。")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/voices/design/save")
def save_voice_design(payload: dict[str, Any]):
    preview_id = str(payload.get("preview_id", ""))
    if not re.fullmatch(r"[a-f0-9]{32}", preview_id):
        raise HTTPException(status_code=422, detail="试听编号无效。")
    manifest_path = VOICE_DESIGN_PREVIEW_DIR / f"{preview_id}.json"
    audio_path = VOICE_DESIGN_PREVIEW_DIR / f"{preview_id}.wav"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="这条试听已不存在，请重新生成。")
    name = str(payload.get("name", "")).strip()
    if not name or len(name) > 36:
        raise HTTPException(status_code=422, detail="音色名称需为 1 至 36 个字符。")
    if _profile_name_taken(name):
        raise HTTPException(status_code=409, detail="这个音色名称已存在。")
    task_id = "voice-save:" + uuid.uuid4().hex
    _claim_task(task_id)
    try:
        before = {item["id"] for item in list_profiles()}
        result = save_voice_profile(name, str(audio_path), manifest["text"], "icl", progress=lambda *args, **kwargs: None)
        created = [item for item in list_profiles() if item["id"] not in before]
        if not created:
            raise HTTPException(status_code=400, detail=re.sub(r"<[^>]*>", "", str(result[1])))
        metadata = created[0]
        folder = _profile_dir(metadata["id"])
        metadata["voice_design"] = {
            "instruction": manifest["instruction"],
            "language": manifest["language"],
            "preview_text": manifest["text"],
        }
        (folder / "profile.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        audio_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        return _profile_payload(metadata)
    finally:
        _release_task(task_id)

@app.get("/api/history")
def get_history():
    items = []
    for path in HISTORY_DIR.glob("*.json"):
        if path.name.startswith("._"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("id") and path.name == f"{record['id']}.json":
                items.append(record)
        except (OSError, ValueError, TypeError):
            continue
    items.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return [_history_payload(record) for record in items[:200]]


@app.get("/api/history/{batch_id}")
def get_history_record(batch_id: str):
    try:
        return _history_payload(_read_history(batch_id))
    except (OSError, ValueError, TypeError):
        raise HTTPException(status_code=404, detail="生成记录不存在。")


@app.get("/api/history/{batch_id}/settings")
def history_settings(batch_id: str):
    try:
        record = _read_history(batch_id)
    except (OSError, ValueError, TypeError):
        raise HTTPException(status_code=404, detail="生成记录不存在。")
    return {
        "voice_mode": record.get("model_mode", "clone"),
        "profile_id": record.get("profile_id"),
        "speaker": record.get("speaker"),
        "instruction": record.get("params", {}).get("instruction", DEFAULT_CUSTOM_VOICE_INSTRUCTION),
        "script": "\n".join(line.get("text", "") for line in record.get("lines", [])),
        "segments": [
            {
                "text": line.get("text", ""),
                **_line_voice(line, record),
            }
            for line in record.get("lines", [])
        ],
        "params": record.get("params", {}),
    }


@app.get("/api/files/{filename}")
def get_audio_file(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="文件不存在。")
    path_string = _safe_output_path(filename)
    if not path_string:
        raise HTTPException(status_code=404, detail="文件不存在。")
    return FileResponse(path_string, media_type="audio/wav")


@app.get("/api/history/{batch_id}/download")
def download_history(batch_id: str):
    try:
        record = _read_history(batch_id)
    except (OSError, ValueError, TypeError):
        raise HTTPException(status_code=404, detail="生成记录不存在。")
    files = _record_files(record)
    if not files:
        raise HTTPException(status_code=404, detail="这批记录没有可下载的音频。")
    zip_path = EXPORT_DIR / f"口播_{batch_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(path).name)
        archive.writestr("generation.json", json.dumps(record, ensure_ascii=False, indent=2))
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.post("/api/jobs")
def create_generation_job(payload: dict[str, Any]):
    voice_mode = payload.get("voice_mode", "clone")
    if voice_mode not in {"clone", "custom_voice"}:
        raise HTTPException(status_code=422, detail="请选择有效的声音来源。")
    if voice_mode == "clone" and not (MODEL_DIR / "model.safetensors").is_file():
        raise HTTPException(status_code=409, detail="Base 权重未就绪。")
    if voice_mode == "custom_voice" and not (CUSTOMVOICE_MODEL_DIR / "model.safetensors").is_file():
        raise HTTPException(status_code=409, detail="CustomVoice 权重未就绪。")
    params = _sampling_params(payload)
    params["language"] = payload.get("language", "Auto")
    if params["language"] not in {value for _, value in LANGUAGES}:
        raise HTTPException(status_code=422, detail="请选择支持的语言。")
    requested = payload.get("segments")
    if isinstance(requested, list):
        raw_segments = requested
    else:
        raw_segments = [{"text": line} for line in str(payload.get("script", "")).splitlines()]
    segments = []
    for item in raw_segments:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            segments.append((item if isinstance(item, dict) else {}, text))
    if not segments:
        raise HTTPException(status_code=422, detail="请先输入要合成的文案。")
    if len(segments) > 8:
        raise HTTPException(status_code=422, detail="每批最多 8 段。")
    if any(len(text) > 4000 for _, text in segments):
        raise HTTPException(status_code=422, detail="单段文案不能超过 4,000 字。")
    global_profile = next((item for item in list_profiles() if item["id"] == payload.get("profile_id")), None)
    global_speaker = payload.get("speaker", "Uncle_Fu")
    if voice_mode == "clone" and global_profile is None:
        raise HTTPException(status_code=422, detail="请先添加并选择一个音色。")
    if voice_mode == "custom_voice" and global_speaker not in CUSTOM_VOICE_LABELS:
        raise HTTPException(status_code=422, detail="请选择有效的官方预制音色。")

    normalized_lines = []
    for index, (segment, text) in enumerate(segments, start=1):
        mode = segment.get("voice_mode") or voice_mode
        if mode not in {"clone", "custom_voice"}:
            raise HTTPException(status_code=422, detail=f"第 {index} 段的声音来源无效。")
        language = segment.get("language") or params["language"]
        if language not in {value for _, value in LANGUAGES}:
            raise HTTPException(status_code=422, detail=f"第 {index} 段的语言无效。")
        if mode == "clone":
            profile_id = segment.get("profile_id") or payload.get("profile_id")
            profile = next((item for item in list_profiles() if item["id"] == profile_id), None)
            if not profile:
                raise HTTPException(status_code=422, detail=f"第 {index} 段的音色不可用。")
            voice = {"mode": mode, "profile_id": profile_id, "speaker": None, "instruction": "", "language": language}
        else:
            speaker = segment.get("speaker") or global_speaker
            if speaker not in CUSTOM_VOICE_LABELS:
                raise HTTPException(status_code=422, detail=f"第 {index} 段的官方音色无效。")
            instruction = segment.get("instruction", payload.get("instruction", DEFAULT_CUSTOM_VOICE_INSTRUCTION))
            if len(str(instruction)) > 1200:
                raise HTTPException(status_code=422, detail=f"第 {index} 段的风格指令超过 1,200 字。")
            voice = {
                "mode": mode,
                "profile_id": None,
                "speaker": speaker,
                "instruction": str(instruction).strip(),
                "language": language,
            }
        normalized_lines.append({
            "index": index,
            "text": text,
            "status": "待生成",
            "output_file": None,
            "error": "",
            "voice": voice,
        })

    if all(line["voice"]["mode"] == "clone" for line in normalized_lines):
        profile_ids = {line["voice"]["profile_id"] for line in normalized_lines}
        profile_name = next((item["name"] for item in list_profiles() if item["id"] == next(iter(profile_ids))), "音色") if len(profile_ids) == 1 else "多种音色"
        record_mode = "clone"
    elif all(line["voice"]["mode"] == "custom_voice" for line in normalized_lines):
        speakers = {line["voice"]["speaker"] for line in normalized_lines}
        profile_name = CUSTOM_VOICE_LABELS[next(iter(speakers))] if len(speakers) == 1 else "多种音色"
        record_mode = "custom_voice"
    else:
        profile_name = "多种音色"
        record_mode = voice_mode

    record = {
        "schema_version": 2,
        "id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_mode": record_mode,
        "profile_id": global_profile["id"] if global_profile else None,
        "profile_name": profile_name,
        "speaker": global_speaker if voice_mode == "custom_voice" else None,
        "params": params | {"instruction": str(payload.get("instruction", "")).strip()},
        "lines": normalized_lines,
    }
    _write_history(record)
    try:
        _launch_api_job(record, [line["index"] for line in normalized_lines])
    except HTTPException:
        for line in record["lines"]:
            line["status"] = "待生成"
        _write_history(record)
        raise
    return _history_payload(record)


@app.get("/api/jobs/{batch_id}")
def get_job(batch_id: str):
    return get_history_record(batch_id)


@app.post("/api/jobs/{batch_id}/stop")
def stop_job(batch_id: str):
    if _active_task() != batch_id:
        raise HTTPException(status_code=409, detail="这批口播当前没有在生成。")
    STOP_REQUESTED.set()
    return {"stopping": True}


@app.post("/api/history/{batch_id}/resume")
def resume_history(batch_id: str):
    try:
        record = _read_history(batch_id)
    except (OSError, ValueError, TypeError):
        raise HTTPException(status_code=404, detail="生成记录不存在。")
    indexes = [int(line["index"]) for line in record["lines"] if line.get("status") != "已完成"]
    if not indexes:
        raise HTTPException(status_code=409, detail="这批口播已全部完成。")
    for line in record["lines"]:
        if line.get("status") != "已完成":
            line["status"] = "待生成"
            line["error"] = ""
    _write_history(record)
    try:
        _launch_api_job(record, indexes)
    except HTTPException:
        raise
    return _history_payload(record)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("QWEN_TTS_PORT", "8000")),
        access_log=False,
        log_level="warning",
    )
