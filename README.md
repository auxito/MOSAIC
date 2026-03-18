# MOSAIC

**M**ulti-agent **O**rchestrated **S**imulation for **A**daptive **I**nterview **C**oaching

---

一个事件驱动的多 Agent 系统，不是聊天机器人包装器，而是一个解决真实多 Agent 挑战的完整架构：**记忆管理、人设一致性、信息不对称、人机切换、可扩展工作流**。

v2 定位：**真人求职者的职业发展教练平台** — 简历优化不篡改事实，面试官深度追问，评估官给出改进建议和学习路线图。

---

## Quick Start

### 1. 安装依赖

```bash
cd MOSAIC
pip install -r requirements.txt
```

依赖清单（全部轻量）：

```
openai          — LLM 调用（兼容任何 OpenAI 格式的 API）
pydantic        — 数据校验
jinja2          — Prompt 模板
rich            — 终端 UI
tiktoken        — Token 计数
python-dotenv   — 环境变量
```

### 2. 配置 API

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```bash
# 任何 OpenAI 兼容 API 均可（OpenAI / DeepSeek / 月之暗面 / 本地 Ollama 等）
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

<details>
<summary>💡 各家 API 配置示例</summary>

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

**本地 Ollama**
```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b
```
</details>

### 3. 运行

```bash
# 进入项目目录后，所有命令都用这个格式：
cd MOSAIC
PYTHONPATH=src python -m mosaic.main [参数]
```

---

## 四种工作流模式

### 🎯 完整面试模拟（推荐首次体验）

简历优化 → AI 面试 → 评估报告，全自动跑完。

```bash
# 默认 10 轮面试
PYTHONPATH=src python -m mosaic.main -w full_interview

# 快速体验：3 轮面试
PYTHONPATH=src python -m mosaic.main -w full_interview -r 3
```

运行后会依次提示你：
1. **选择简历输入** — 输入 `sample` 使用内置示例简历，或 `text` 手动粘贴
2. **输入 JD** — 粘贴目标职位描述，或回车使用默认
3. **选择简历风格** — 三选一（见下方说明）
4. 自动进入面试 → 评估 → 生成报告

### ✏️ 仅简历优化

只做简历优化，不进行面试，适合快速查看三种风格的效果。

```bash
PYTHONPATH=src python -m mosaic.main -w resume_only
```

### 🎤 真人练习模式

面试官提问，**你来回答**。适合真实面试准备。

```bash
PYTHONPATH=src python -m mosaic.main -w human_practice -r 5
```

### 📊 仅面试评估

跳过简历修改，直接面试 + 评估。

```bash
PYTHONPATH=src python -m mosaic.main -w eval_only -r 5
```

---

## 三种简历优化风格

| 风格 | 名称 | 温度 | 说明 |
|------|------|------|------|
| `rigorous` | 专业打磨版 | 0.3 | **零虚构**，仅优化表达 + STAR 格式 + JD 对齐。附修改说明和学习建议 |
| `embellished` | 技术深化版 | 0.5 | 基于已有方向做**技术深度扩展**（如"用 Redis"→"多级缓存架构"）。附学习建议 + 面试准备要点 |
| `wild` | 成长路线版 | 0.7 | **大胆扩展到前沿技术**（如"Docker 部署"→"K8s + Service Mesh"）。附完整学习路线图（按周规划） |

**核心原则**：三种风格都**不会篡改基本事实**（教育背景、公司名、头衔、时间、数字）。区别在于技术深度的扩展程度。

---

## 面试过程中会发生什么

```
┌─────────────────────────────────────────────────────────┐
│                     面试循环                              │
│                                                         │
│   面试官提问 ──→ 候选人回答 ──→ 评估官打分+建议           │
│       ↑              │              │                    │
│       │              │    ┌─────────┘                    │
│       │              │    │                              │
│  根据评分调整策略  记忆管理器抽取事实/检测矛盾             │
│  🟢 答得好→深入    如发现矛盾→通知面试官追问              │
│  🟡 一般→引导                                            │
│  🔴 答不上→换话题                                        │
└─────────────────────────────────────────────────────────┘
```

**面试官**是领域技术专家，会沿着简历关键词深挖到技术纵深：
- "了解 Transformer" → self-attention → multi-head attention → 位置编码对比
- "使用 Redis" → 缓存策略 → 一致性问题 → 缓存穿透/雪崩

**评估官**每轮给出：
- 四维评分（质量/技术/沟通/可信度）
- ✨ 亮点、💡 改进建议、📚 知识盲区

**最终报告**包含：
- 综合评分 + 五维雷达
- 教练建议
- 📚 个性化学习路线图（按领域、按周规划、附推荐资源）

报告自动保存到 `reports/` 目录。

---

## 运行测试

```bash
cd MOSAIC
python -m pytest tests/ -v
# 42 tests — 覆盖记忆系统、事件总线、工作流、信息不对称、集成测试
```

---

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                       │
│              (State Machine + Event Bus)              │
│                                                      │
│  Phase: INIT → RESUME_INPUT → RESUME_MODIFY          │
│         → INTERVIEW_LOOP → EVALUATION → REPORT       │
└──────────────────────────────────────────────────────┘
        ↕ Event Bus ↕
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐
│ Interviewer│ │Interviewee │ │ Evaluator  │ │MemoryManager │ │CareerCoach │
│ 领域专家   │ │双简历候选人 │ │ 教练评估官 │ │事实抽取      │ │ 3 种风格   │
│ 自适应策略 │ │参考答案级  │ │ 改进建议   │ │矛盾检测      │ │ 学习路线   │
└────────────┘ └────────────┘ └────────────┘ └──────────────┘ └────────────┘
```

### 四层记忆系统

| 层 | 职责 | 解决的问题 |
|----|------|-----------|
| **Privileged** | 按 Agent 控制信息可见性 (MemoryPolicy) | 信息不对称 — 评估官看两份简历，面试者看两份 + JD |
| **Semantic** | 从对话中抽取结构化事实 | 人设一致性 — 面试者回答前检查已知事实 |
| **Working** | 动态组合上下文窗口 | Token 预算 — 渐进式摘要控制上下文长度 |
| **Episodic** | 只追加的完整对话日志 | 回放和评估的原始数据 |

### 核心机制

Agent 之间通过事件总线通信，不直接耦合：
- MemoryManager 发现矛盾 → 发布 `CONTRADICTION_FOUND`
- Interviewer 收到 → 下一轮温和追问矛盾点
- Evaluator 收到 → 记录为评估依据
- Evaluator 发布 `TURN_EVALUATED` → Interviewer 收到 → 动态调整追问策略

这是**涌现行为**，不是硬编码逻辑。

---

## 项目结构

```
MOSAIC/
├── src/mosaic/
│   ├── core/                 # 可复用框架层
│   │   ├── events.py         # EventType + EventBus
│   │   ├── agent.py          # BaseAgent ABC + MemoryPolicy
│   │   ├── orchestrator.py   # 状态机 + 事件路由
│   │   ├── workflow.py       # Phase 枚举 + WorkflowConfig
│   │   └── memory/           # 四层记忆系统
│   ├── agents/               # 6 个领域 Agent
│   │   ├── career_coach.py   # 职业发展教练（简历优化）
│   │   ├── interviewer.py    # 领域专家面试官
│   │   ├── interviewee.py    # 双简历候选人
│   │   ├── evaluator.py      # 教练型评估官
│   │   ├── memory_manager.py # 记忆管理器
│   │   └── resume_parser.py  # 简历解析器
│   ├── participants/         # 人机切换（Protocol 模式）
│   ├── prompts/              # Jinja2 模板（6 个）
│   ├── workflows/            # 阶段处理器注册
│   ├── resume/               # 简历数据模型 + 解析
│   ├── llm/                  # OpenAI SDK 封装
│   └── output/               # Markdown 报告生成
├── tests/                    # 42 个单元 + 集成测试
└── reports/                  # 生成的面试报告
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
  -v, --verbose         详细日志
```

### 常用命令

```bash
# 快速体验（3轮，使用示例简历）
PYTHONPATH=src python -m mosaic.main -w full_interview -r 3

# 认真练习（5轮，自己回答）
PYTHONPATH=src python -m mosaic.main -w human_practice -r 5

# 只看简历优化效果
PYTHONPATH=src python -m mosaic.main -w resume_only

# 详细日志模式
PYTHONPATH=src python -m mosaic.main -w full_interview -r 3 -v
```

---

## .env 配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | API Key（必填） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 地址 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 模型名 |
| `TOKEN_BUDGET` | `12000` | 上下文窗口 token 预算 |
| `VERBATIM_WINDOW` | `6` | 保留最近几轮原文对话 |
| `SUMMARY_BATCH` | `5` | 每多少轮触发一次摘要 |
| `DEFAULT_ROUNDS` | `10` | 默认面试轮次 |
| `REPORT_DIR` | `reports` | 报告输出目录 |
