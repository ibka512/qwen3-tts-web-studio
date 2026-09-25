# Qwen3-TTS 本地语音工作台

一个运行在本机的 Qwen3-TTS Base 网页界面，围绕音色克隆和批量语音生成功能构建。模型权重不包含在本仓库中。

## 功能

- 创建和复用音色档案，试听参考音频
- 支持 ICL（参考音频加逐字稿）和仅提取音色两种克隆方式
- 选择语音语言、调整采样参数并应用快速预设
- 按行生成 WAV，查看进度，停止后继续未完成段落
- 查看历史批次、试听与下载结果、载入原始设置
- 在回收站恢复误移除的音色档案

## 本地运行

需要 Python 3.12 和已下载的 `Qwen3-TTS-12Hz-1.7B-Base` 模型目录。请先按硬件平台准备 PyTorch，再安装其余依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

默认情况下，程序会在源码目录下查找模型和保存本地数据。若模型和数据放在其他位置，设置 `QWEN_TTS_ROOT` 指向数据根目录；模型目录默认位于该目录的 `models/Qwen3-TTS-12Hz-1.7B-Base`。也可通过 `QWEN_TTS_MODEL_DIR` 单独指定模型目录。

```bash
export QWEN_TTS_ROOT="/path/to/qwen3-tts-data"
python web_app.py
```

页面默认只监听 `127.0.0.1:8000`，不会公开到互联网。模型、音色档案、生成记录和音频文件需由使用者自行准备或生成，均保存在 `QWEN_TTS_ROOT` 指定的数据根目录中。

## 目录说明

- `models/`：模型权重（不提交到 Git）
- `voices/`：音色档案与可恢复的回收站内容
- `outputs/`：生成的 WAV 文件
- `history/`：批次文本、参数和逐段状态
- `exports/`：导出的音色档案 ZIP
- `logs/`：本机运行日志

这些运行数据默认被 `.gitignore` 排除，避免把模型权重、参考音频、生成文本或语音文件推送到远端。
