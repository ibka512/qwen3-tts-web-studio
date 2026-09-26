# Qwen3-TTS 本地语音工作台

一个面向 Apple Silicon Mac 的本地网页工作台，把 Qwen3-TTS 的参考音频克隆和官方预制音色合在同一个界面中。模型在本机运行，网页默认只监听 `127.0.0.1`。

模型权重、音色参考音频、生成语音和历史记录均不包含在本仓库中。

## 功能

- **音色克隆**：建立和管理音色档案，支持 ICL（参考音频加逐字稿）和仅提取音色两种方式
- **官方预制音色**：选择 Vivian、Serena、Uncle Fu、Dylan、Eric、Ryan、Aiden、Ono Anna 或 Sohee
- **风格指令**：为官方预制音色提供语气、语速和表达方式提示
- **逐段生成**：按行合成 WAV，查看进度，停止后继续或重试未完成段落
- **生成记录**：试听和下载结果，查看文本与设置，并将设置载回工作台
- **本地保存**：音色档案、生成音频和历史记录保存在指定的数据目录
- **单模型驻留**：克隆和预制音色模型按使用模式轮流加载，避免同时占用内存

## 环境要求

- Apple Silicon Mac
- Python 3.12
- 支持 MPS 的 PyTorch
- 本地 Qwen3-TTS Base 模型；使用官方预制音色时还需 CustomVoice 模型

## 安装与启动

先按 PyTorch 官方说明准备适用于当前 macOS 的 PyTorch，然后安装项目依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认情况下，程序会在源码目录下查找模型并保存运行数据。建议把权重与运行数据放到独立目录，再通过 `QWEN_TTS_ROOT` 指定：

```bash
export QWEN_TTS_ROOT="$HOME/qwen3-tts-data"
python web_app.py
```

默认目录结构如下：

```text
qwen3-tts-data/
├── models/
│   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   └── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── voices/
├── outputs/
├── history/
├── exports/
└── logs/
```

Base 模型在启动时载入。CustomVoice 模型在首次使用“官方预制音色”模式时载入；首次载入或切换模型需要等待一段时间。只使用音色克隆时可以不准备 CustomVoice 模型。

如需自定义位置，可设置以下环境变量：

| 变量 | 用途 | 默认位置 |
| --- | --- | --- |
| `QWEN_TTS_ROOT` | 音色档案、输出、历史记录、日志与默认模型目录 | 源码目录 |
| `QWEN_TTS_MODEL_DIR` | Base 模型目录 | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base` |
| `QWEN_TTS_CUSTOM_MODEL_DIR` | CustomVoice 模型目录 | `$QWEN_TTS_ROOT/models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `QWEN_TTS_PORT` | 本地网页端口 | `8000` |

模型使用本地文件载入。准备好依赖和权重后，运行：

```bash
python web_app.py
```

然后访问 <http://127.0.0.1:8000/>。

## 本地数据与隐私

- `voices/`：音色档案、参考音频和回收站
- `outputs/`：生成的 WAV 文件
- `history/`：每批文本、参数和逐段生成状态
- `exports/`：导出的音色档案 ZIP
- `logs/`：本机运行日志

这些运行数据目录及模型权重由 `.gitignore` 排除。请勿将私人参考音频、生成文本或生成语音提交到仓库。
