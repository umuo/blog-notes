# OpenClaw 与 Hermes 退款会话对照

使用 Node.js 24 和 Python 3.14。Chat Completions 夹具只监听本机回环，读取固定的合成 CSV。OpenClaw 状态、Hermes profile 和日志在临时目录中创建并清理。没有模型权重、真实 API 密钥或消息渠道。

## OpenClaw

在仓库根目录执行以下命令。把 npm 包安装在临时目录，不影响现有 OpenClaw 配置。

```bash
case_root=$(mktemp -d)
npm install --prefix "$case_root/open" --no-audit --no-fund openclaw@2026.11.7
python3 experiments/framework-replay/replay.py --openclaw "$case_root/open/node_modules/openclaw/openclaw.mjs"
```

## Hermes Agent

检出 `v0.23.3`。本次验证在该标签源码上直接运行 `AIAgent`，安装退款回放所需的运行依赖。完整运行依赖在本机对浏览器包和 Pillow 的版本约束产生冲突，因此没有启用浏览器、图像处理和文档抽取模块。

```bash
case_root=$(mktemp -d)
git clone --depth 1 --branch v0.23.3 https://github.com/NousResearch/hermes-agent.git "$case_root/hermes"
uv venv --python 3.14 "$case_root/venv"
python3 - "$case_root/hermes/pyproject.toml" > "$case_root/requirements.txt" <<'PY'
import sys
import tomllib
from pathlib import Path

metadata = tomllib.loads(Path(sys.argv[1]).read_text())
for requirement in metadata["project"]["dependencies"]:
    if not requirement.lower().startswith(("browser-harness", "pillow", "firecrawl-anydoc")):
        print(requirement)
PY
uv pip install --python "$case_root/venv/bin/python" -r "$case_root/requirements.txt"
PYTHONPATH="$case_root/hermes" "$case_root/venv/bin/python" experiments/framework-replay/replay.py --hermes
```

观察 `turns` 的六步结果、规则来源和工具名。Hermes 审批通过本版本的命令处理函数调用；OpenClaw 则在夹具中人为更新工作区笔记。实验对照业务读动作与状态加载，渠道接入和后台回顾依据版本固定的源码解读。
