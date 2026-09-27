# Qwen3-TTS 本地语音工作台

一个面向 Apple Silicon Mac 的本地语音创作网页，提供音色克隆、官方预制音色和文字设计音色。推理由本机运行，网页只绑定 `127.0.0.1`；音色档案、生成记录和音频也保存在本机。

## 功能

- **音色克隆**：用参考录音和对应原文（ICL），或只用参考录音（x-vector）创建音色。
- **VoiceDesign**：用文字描述声音，生成试听，并保存为可复用的音色档案。
- **CustomVoice**：使用 Qwen3-TTS 的 9 个官方预制音色，并添加风格指令。
- **分段口播**：按换行分段；最多 8 段，每段最多 4,000 字；可逐段选择声音和语言。
- **生成参数**：调整主采样、声音细节采样和单段长度上限。
- **任务控制**：查看逐段状态，暂停后继续未完成段，或重试失败段。
- **音频与记录**：试听、下载单段或整批 WAV；回看文案与参数。
- **音色管理**：上传或录制参考音频、重命名、导出、移入回收站和恢复音色。
- **界面**：中文、黏土拟物风格、浅色和深色模式，支持桌面与手机屏幕。

音频按段生成，单段完成后即可试听和下载；此版本不提供实时流式播放。

## 环境要求

- Apple Silicon Mac
- Python 3.12
- 按 PyTorch 官方安装说明准备支持 MPS 的 PyTorch
- Base 模型权重（必需）；CustomVoice 和 VoiceDesign 权重按需安装

## 安装与启动

```bash
git clone https://github.com/ibka512/qwen3-tts-web-studio.git
cd qwen3-tts-web-studio
python3.12 -m venv .venv
source .venv/bin/activate
```

先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装适用于当前 macOS 的 PyTorch，再安装项目依赖并启动：

```bash
python -m pip install -r requirements.txt
export QWEN_TTS_ROOT="$HOME/qwen3-tts-data"
python web_app.py
```

打开 <http://127.0.0.1:8000/>。默认端口为 `8000`，可用 `QWEN_TTS_PORT` 修改。首次启动会载入 Base 模型；首次切换到其他声音功能时会载入对应模型。为控制统一内存占用，同一时间只在内存中保留一个模型。

前端使用原生 HTML、CSS 和 JavaScript，无需安装 Node.js 或运行前端构建命令。

## 模型文件

下载完整模型仓库，并将文件放入下面对应的目录。每个模型目录中应直接包含 `model.safetensors` 和 `speech_tokenizer/` 子目录。

| 功能 | 官方模型 | 本地目录 |
| --- | --- | --- |
| 音色克隆（必需） | [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base` |
| 官方预制音色 | [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| 文字设计音色 | [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |

可以从模型页面下载，也可以安装 Hugging Face Hub 命令行工具后使用 `hf download`：

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir "$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base"
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir "$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local-dir "$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
```

只使用某项功能时，可只下载 Base 和该功能所需的模型。Base 权重缺失时工作台无法启动；未安装 CustomVoice 或 VoiceDesign 权重时，对应功能不可用。

## 文件位置与配置

默认情况下，模型和运行数据都放在源码目录下。设置 `QWEN_TTS_ROOT` 后，模型、运行缓存和创作数据改存到指定目录：

```text
$QWEN_TTS_ROOT/
├── models/       # 模型权重
├── voices/       # 音色档案和参考录音
├── outputs/      # 生成的 WAV
├── history/      # 文案、设置和生成状态
├── exports/      # 导出的音色档案
├── logs/         # 运行日志
└── cache/        # 临时文件和运行缓存
```

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `QWEN_TTS_ROOT` | 模型、音色、输出、历史、导出和缓存的根目录 | 源码目录 |
| `QWEN_TTS_MODEL_DIR` | Base 模型目录 | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base` |
| `QWEN_TTS_CUSTOM_MODEL_DIR` | CustomVoice 模型目录 | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `QWEN_TTS_VOICEDESIGN_MODEL_DIR` | VoiceDesign 模型目录 | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| `QWEN_TTS_PORT` | 本地网页端口 | `8000` |

模型权重、参考录音、生成文案、音色档案和生成音频不包含在本仓库中。不要将私人素材提交到公开仓库。
