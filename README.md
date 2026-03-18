<p align="center">
  <img src="assets/banner.svg" alt="MOSAIC Banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-OpenAI_Compatible-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/UI-Gradio-FF6F00?style=for-the-badge&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Arch-Multi--Agent-00C853?style=for-the-badge&logo=robot&logoColor=white" />
</p>

<p align="center">
  <i>6 个 AI Agent 协作的面试教练平台 — 简历优化 × 深度模拟面试 × 教练级评估反馈，一站式备战求职。</i>
</p>

---

## What is MOSAIC?

> **一句话**：上传简历 → AI 帮你优化 → 模拟真实技术面试 → 生成教练级评估报告 + 个性化学习路线图。

MOSAIC 是一个**事件驱动的多 Agent 面试教练平台**，由 6 个专业 Agent 协作完成从简历优化到面试模拟到评估反馈的全流程。它解决了多 Agent 系统中的真实工程挑战：

- **记忆管理** — 四层记忆系统（特权 / 语义 / 工作 / 情景），渐进式摘要控制上下文长度
- **信息不对称** — 评估官能看两份简历（原版 + 优化版），面试者看不到评分
- **人设一致性** — 语义记忆抽取事实，矛盾检测防止候选人前后不一致
- **人机无缝切换** — AI 托管 / 亲自作答，同一套流程两种体验
- **可扩展工作流** — Phase 枚举 + 工作流配置，新增模式不改 Agent 代码

---

## Features at a Glance

```
                    ┌─── Tab 1: 简历优化 ──────────────────────────────┐
                    │                                                   │
                    │  📎 上传 PDF/DOCX/TXT → LLM 结构化解析           │
                    │  📝 可编辑表单（个人信息/教育/工作/项目/技能）      │
                    │  🎨 三种优化风格 + 教练笔记                       │
                    │                                                   │
                    ├─── Tab 2: 模拟面试 ──────────────────────────────┤
                    │                                                   │
                    │  🤖 AI 托管 / ✍️ 亲自作答                        │
                    │  💬 实时对话 + 翻页评分卡片                       │
                    │  🧠 矛盾检测 + 自适应追问                        │
                    │                                                   │
                    ├─── Tab 3: 评估报告 ──────────────────────────────┤
                    │                                                   │
                    │  📊 综合评分 + 维度分析                           │
                    │  📚 个性化学习路线图                              │
                    │  ⬇️ 一键下载 Markdown 报告                       │
                    └───────────────────────────────────────────────────┘
```

---

## Quick Start — 3 步跑起来

### Step 1: 克隆 & 安装

```bash
git clone https://github.com/your-username/MOSAIC.git
cd MOSAIC

# 推荐使用虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

<details>
<summary>📦 依赖清单（全部轻量，无 GPU 要求）</summary>

| 包 | 用途 |
|---|---|
| `openai` | LLM 调用（兼容任何 OpenAI 格式 API） |
| `pydantic` | 数据校验 |
| `jinja2` | Prompt 模板引擎 |
| `rich` | 终端富文本 UI |
| `tiktoken` | Token 计数 |
| `python-dotenv` | 环境变量管理 |
| `gradio` | Web UI |
| `pdfplumber` | PDF 解析 |
| `python-docx` | Word 文档解析 |

</details>

### Step 2: 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

> **任何 OpenAI 兼容 API 都行** — OpenAI / DeepSeek / 月之暗面 / 通义千问 / 本地 Ollama，只要支持 `/v1/chat/completions` 接口。

<details>
<summary>💡 各家 API 配置示例（点击展开）</summary>

**DeepSeek**
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

**月之暗面 (Kimi)**
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_MODEL=moonshot-v1-8k
```

**通义千问 (DashScope)**
```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
```

**本地 Ollama（免费，无需 API Key）**
```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b
```

</details>

### Step 3: 启动！

```bash
# 🌐 Web UI（推荐，可视化操作）
PYTHONPATH=src python -m mosaic.main --web
# 浏览器打开 http://localhost:7860

# 🖥️ 命令行模式
PYTHONPATH=src python -m mosaic.main -w full_interview -r 3
```

> **Windows 用户**：将 `PYTHONPATH=src` 替换为 `set PYTHONPATH=src &&`，或使用 PowerShell：`$env:PYTHONPATH="src"; python -m mosaic.main --web`

---

## 四种工作流模式

| 模式 | 命令 | 说明 | 适合场景 |
|------|------|------|---------|
| **完整面试** | `-w full_interview` | 简历优化 → 模拟面试 → 评估报告 | 首次体验 / 全流程演练 |
| **仅简历优化** | `-w resume_only` | 三种风格简历优化 + 教练笔记 | 快速优化简历 |
| **真人练习** | `-w human_practice` | 面试官提问，你来回答 | 真实面试备战 |
| **仅评估** | `-w eval_only` | 跳过简历修改，直接面试 + 评估 | 测试面试水平 |

```bash
# 完整面试（3 轮快速体验）
PYTHONPATH=src python -m mosaic.main -w full_interview -r 3

# 真人练习（5 轮，自己回答面试问题）
PYTHONPATH=src python -m mosaic.main -w human_practice -r 5

# 只看简历优化效果
PYTHONPATH=src python -m mosaic.main -w resume_only

# Web UI（推荐）
PYTHONPATH=src python -m mosaic.main --web
```

---

## 三种简历优化风格

| 风格 | 名称 | 温度 | 策略 |
|------|------|------|------|
| `rigorous` | **专业打磨版** | 0.3 | **零虚构**，仅优化表达 + STAR 格式 + JD 对齐。附修改说明 |
| `embellished` | **技术深化版** | 0.5 | 基于已有方向做技术深度扩展（如 "用 Redis" → "多级缓存架构"）。附面试准备要点 |
| `wild` | **成长路线版** | 0.7 | 大胆扩展到前沿技术（如 "Docker 部署" → "K8s + Service Mesh"）。附完整学习路线图 |

> **核心原则**：三种风格都**不会篡改基本事实**（教育背景、公司名、头衔、时间线、数字），区别在于技术深度的扩展程度。

---

## 面试过程中发生了什么？

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                        面试循环                                   │
    │                                                                  │
    │    面试官提问 ───→ 候选人回答 ───→ 评估官打分 + 建议              │
    │        ↑               │                │                        │
    │        │               ↓                ↓                        │
    │   根据评分调整      记忆管理器       记录事实                      │
    │   追问策略          抽取事实         检测矛盾                      │
    │                     检测矛盾                                      │
    │                        │                                         │
    │   🟢 答得好 → 深入追问               如发现矛盾                   │
    │   🟡 一般   → 引导回答               → 通知面试官温和追问          │
    │   🔴 答不上 → 换话题                                              │
    └──────────────────────────────────────────────────────────────────┘
```

**面试官**是领域技术专家，沿简历关键词深挖到技术纵深：
- "了解 Transformer" → self-attention → multi-head attention → 位置编码对比
- "使用过 Redis" → 缓存策略 → 一致性问题 → 缓存穿透/雪崩解法

**评估官**每轮给出四维评分 + 亮点 / 改进建议 / 知识盲区。

**最终报告**包含综合评分、教练建议、个性化学习路线图（按领域分类、附推荐资源）。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        Orchestrator                           │
│                  (State Machine + Event Bus)                   │
│                                                               │
│   Phase: INIT → RESUME_INPUT → RESUME_MODIFY                 │
│          → INTERVIEW_LOOP → EVALUATION → REPORT → COMPLETE    │
└──────────────────────────────────────────────────────────────┘
                         ↕ Event Bus ↕
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐ ┌─────────────┐ ┌─────────────┐
│ CareerCoach │ │ Interviewer │ │ Interviewee │ │MemoryManager  │ │  Evaluator  │ │ResumeParser │
│ 3 种风格    │ │ 领域专家    │ │ 双简历感知  │ │ 事实抽取      │ │ 教练评估    │ │ PDF/DOCX    │
│ 教练笔记    │ │ 自适应策略  │ │ 人机切换    │ │ 矛盾检测      │ │ 学习路线    │ │ LLM 结构化  │
└─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘ └─────────────┘ └─────────────┘
```

### 四层记忆系统

| 层 | 职责 | 解决的问题 |
|----|------|-----------|
| **Privileged Memory** | 按 Agent 控制信息可见性 | 信息不对称 — 评估官看两份简历，面试者只看优化版 |
| **Semantic Memory** | 从对话中抽取结构化事实 | 人设一致性 — 检查已知事实防止矛盾 |
| **Working Memory** | 动态组合上下文窗口 | Token 预算 — 渐进式摘要控制上下文长度 |
| **Episodic Memory** | 只追加的完整对话日志 | 回放评估 — 为最终报告保留原始数据 |

### 事件驱动协作

Agent 之间通过事件总线通信，不直接耦合 — 这是**涌现行为**，不是硬编码逻辑：

```
MemoryManager 发现矛盾 → 发布 CONTRADICTION_FOUND
    → Interviewer 收到 → 下一轮温和追问矛盾点
    → Evaluator 收到 → 记录为评估依据

Evaluator 发布 TURN_EVALUATED
    → Interviewer 收到 → 动态调整追问策略（深挖 / 引导 / 换题）
```

---

## 项目结构

```
MOSAIC/
├── src/mosaic/
│   ├── main.py                  # CLI / Web 入口
│   ├── web_app.py               # Gradio Web UI（三 Tab 布局）
│   ├── config.py                # 环境变量配置
│   │
│   ├── core/                    # ===== 可复用框架层 =====
│   │   ├── agent.py             # BaseAgent ABC + MemoryPolicy
│   │   ├── events.py            # EventType 枚举 + EventBus
│   │   ├── orchestrator.py      # 状态机 + 事件路由
│   │   ├── workflow.py          # Phase 枚举 + WorkflowConfig
│   │   └── memory/              # 四层记忆系统
│   │       ├── privileged.py    # 特权记忆（信息不对称）
│   │       ├── semantic.py      # 语义记忆（事实抽取）
│   │       ├── working.py       # 工作记忆（上下文窗口）
│   │       ├── episodic.py      # 情景记忆（对话日志）
│   │       └── context_manager.py  # Token 预算管理
│   │
│   ├── agents/                  # ===== 6 个领域 Agent =====
│   │   ├── career_coach.py      # 职业发展教练（简历优化）
│   │   ├── interviewer.py       # 领域专家面试官
│   │   ├── interviewee.py       # 双简历候选人
│   │   ├── evaluator.py         # 教练型评估官
│   │   ├── memory_manager.py    # 记忆管理器
│   │   └── resume_parser.py     # 简历解析器
│   │
│   ├── participants/            # 人机切换（Protocol 模式）
│   │   ├── protocol.py          # Participant Protocol 定义
│   │   ├── ai_participant.py    # AI 自动回答
│   │   └── human_participant.py # 真人输入
│   │
│   ├── prompts/                 # Jinja2 Prompt 模板
│   │   ├── interviewer_system.j2
│   │   ├── interviewee_system.j2
│   │   ├── evaluator_system.j2
│   │   ├── resume_rigorous.j2   # 专业打磨版
│   │   ├── resume_embellished.j2 # 技术深化版
│   │   └── resume_wild.j2      # 成长路线版
│   │
│   ├── workflows/               # 阶段处理器
│   │   ├── full_interview.py
│   │   ├── resume_only.py
│   │   └── human_practice.py
│   │
│   ├── resume/                  # 简历数据层
│   │   ├── schema.py            # ResumeData / Education / WorkExperience Pydantic 模型
│   │   ├── file_parser.py       # PDF/DOCX/TXT 解析 + LLM 结构化
│   │   └── structured_input.py  # 结构化输入辅助
│   │
│   ├── llm/
│   │   └── client.py            # AsyncOpenAI 封装（chat / stream / json）
│   │
│   └── output/
│       └── report.py            # Markdown 报告生成
│
├── tests/                       # 42 个单元 + 集成测试
│   ├── unit/
│   │   ├── test_agents/
│   │   ├── test_memory/
│   │   └── test_file_parser.py
│   └── integration/
│       └── test_integration.py
│
├── examples/                    # 示例数据
│   ├── sample_resume.json
│   └── sample_jd.md
│
├── reports/                     # 生成的面试报告（自动创建）
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## CLI 参数速查

```bash
PYTHONPATH=src python -m mosaic.main [选项]

选项:
  -w, --workflow {full_interview,resume_only,human_practice,eval_only}
                        工作流模式 (默认: full_interview)
  -r, --rounds N        面试轮次 (默认: 10)
  --human               真人模式（你来回答面试问题）
  --web                 启动 Gradio Web UI (http://localhost:7860)
  -v, --verbose         详细日志
```

---

## .env 完整配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | **必填**，API Key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 地址 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 模型名 |
| `TOKEN_BUDGET` | `12000` | 上下文窗口 Token 预算 |
| `VERBATIM_WINDOW` | `6` | 保留最近几轮原文对话数 |
| `SUMMARY_BATCH` | `5` | 每多少轮触发一次渐进式摘要 |
| `DEFAULT_ROUNDS` | `10` | 默认面试轮次 |
| `REPORT_DIR` | `reports` | 报告输出目录 |

---

## 测试

```bash
cd MOSAIC
python -m pytest tests/ -v
# 42 tests — 覆盖记忆系统、事件总线、工作流、信息不对称、集成测试
```

---

## Tech Stack

| 领域 | 技术 |
|------|------|
| LLM 调用 | OpenAI SDK（兼容任何 `/v1/chat/completions` API） |
| 数据模型 | Pydantic v2 |
| Prompt 工程 | Jinja2 模板 |
| Web UI | Gradio |
| 文件解析 | pdfplumber + python-docx |
| Token 管理 | tiktoken |
| CLI UI | Rich |
| 测试 | pytest + pytest-asyncio |

---

## License

MIT

---

<p align="center">
  <b>MOSAIC</b> — 让每一次面试练习都有教练级的反馈。
</p>
