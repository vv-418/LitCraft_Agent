# LitCraft Agent

LitCraft Agent 是一个从 0 开始实现的智能文献综述 Agent 项目，支持从学术搜索引擎检索论文、下载 PDF、解析全文、向量存储、语义检索，最终自动生成文献综述。

## 快速开始

### 1. 配置环境变量

创建 `.env` 文件（或从 `.env.example` 复制），填入以下内容：

```env
LLM_MODEL_ID=你的模型名
LLM_API_KEY=你的 API Key
LLM_BASE_URL=你的 OpenAI-compatible 接口地址
LLM_TIMEOUT=60
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
# 基本用法
python main.py --topic "你的研究主题"

# 限制文献年份（只搜索 2020 年及之后的文献）
python main.py --topic "Transformer 在 NLP 中的应用" --year-from 2020

# 启用 PDF 保存（指定输出路径）
python main.py --topic "RAG 在医学问答中的应用" --save-pdf output/review.pdf

#专用python路径
d:\Users\vv\Anaconda3\envs\litcraft\python.exe
```

### Web API 方式（需额外启动后端）

```bash
# 终端1：启动 FastAPI 后端（端口 8000）
python -m uvicorn api.server:app --reload --port 8000

# 终端2：启动 Streamlit 前端（端口 8501）
python -m streamlit run frontend/app.py --server.port 8501
```

然后浏览器打开 http://localhost:8501 即可使用图形界面。

## 目录结构

```text
LitCraft_Agent/
├── main.py                    # 命令行入口
├── llm_client.py              # 大模型调用封装
├── requirements.txt
├── README.md
├── QUICK_START.md
│
├── agent/
│   ├── __init__.py
│   ├── models.py              # 数据结构定义（AgentStep, AgentResult）
│   ├── prompts.py             # 提示词模板
│   └── langgraph_agent.py     # LangGraph ReAct Agent 主循环
│
├── tools/
│   ├── __init__.py
│   ├── base.py                # Tool / ToolSpec 基类
│   ├── registry.py            # 工具注册表
│   ├── paper_downloader.py    # PDF 下载
│   ├── pdf_parser.py          # PDF 文本解析
│   ├── pdf_generator.py       # 综述 PDF 生成
│   ├── text_chunker.py        # 长文本分块
│   ├── vector_store.py        # Chroma 向量存储
│   ├── advanced_retrieval.py  # HyDE + MQE 智能检索
│   │
│   └── search/
│       ├── __init__.py
│       ├── arxiv_search.py          # arXiv API 搜索
│       ├── google_scholar.py        # Google Scholar 搜索（via scholarly）
│       ├── semantic_scholar.py      # Semantic Scholar API 搜索
│       └── multi_source_search.py   # 多源聚合搜索
│
├── output/                   # 输出目录（生成的 PDF、中间文件）
├── storage/
│   └── chroma/               # Chroma 向量数据库持久化目录
│
├── api/                      # FastAPI 后端服务
│   ├── __init__.py
│   ├── schemas.py            # 请求/响应数据结构
│   ├── task_manager.py       # 后台任务管理器
│   └── server.py             # FastAPI 路由入口
│
├── frontend/
│   └── app.py                # Streamlit 前端页面
│
├── output/                   # 输出目录（生成的 PDF、中间文件）
├── storage/
│   └── chroma/               # Chroma 向量数据库持久化目录
│
└── models/                   # 本地 NLP 模型缓存（用于 sentence-transformers）
```

## 脚本说明

### main.py

作用：项目的命令行入口。

它负责：

- 读取用户传入的研究主题及可选参数
- 创建大模型客户端
- 注册所有工具
- 创建 LangGraphAgent
- 运行 Agent
- 打印每一步思考、行动、观察和最终答案

**命令行参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--topic` | str | 必填 | 研究主题 |
| `--year-from` | str | — | 限制文献的最早年份（如 2020） |
| `--save-pdf` | str | — | 将最终答案保存为 PDF 的路径（如 `output/review.pdf`） |

**函数：**

```python
build_agent() -> LangGraphAgent
```

- 从 `main()` 中读取已解析的命令行参数
- 创建 `LitCraftAgentsLLM` 大模型客户端
- 创建 `ToolRegistry` 并注册所有真实工具
- 返回组装好的 `LangGraphAgent` 实例（`max_steps=20`）

```python
main() -> None
```

- 使用 `argparse` 解析命令行参数
- 调用 `build_agent()`
- 执行 `agent.run(args.topic, year_from=args.year_from)`
- 打印 Agent 执行轨迹和最终答案
- 如果指定了 `--save-pdf`，则将最终答案导出为 PDF

### llm_client.py

作用：封装大模型调用。

这个文件让项目内部不用到处直接写 OpenAI/DeepSeek API 调用代码。以后如果换模型，只需要优先改这一层。

类：

```python
class LitCraftAgentsLLM:
```

作用：

- 从 `.env` 读取模型名、API Key、接口地址、超时时间
- 创建 OpenAI-compatible 客户端
- 提供统一的大模型调用方法

改名说明：虽然类仍然叫 `LitCraftAgentsLLM`，但这是基于项目名称 LitCraft

方法：

```python
__init__(...)
```

作用：

- 读取 `LLM_MODEL_ID`
- 读取 `LLM_API_KEY`
- 读取 `LLM_BASE_URL`
- 读取 `LLM_TIMEOUT`
- 初始化 OpenAI-compatible 客户端

```python
think(messages, temperature=0) -> str
```

作用：

- 接收完整 messages 列表
- 调用大模型
- 返回模型输出文本

```python
chat(system_prompt, user_prompt, temperature=0) -> str
```

作用：

- 把 system prompt 和 user prompt 组合成 messages
- 调用 `think()`
- 返回模型输出文本

### agent/models.py

作用：定义 Agent 运行过程中的数据结构。

类：

```python
AgentStep
```

作用：

- 记录 Agent 的单步执行信息

字段：

- `thought`：这一轮 Agent 的思考
- `action`：这一轮要调用的工具名
- `action_input`：工具输入参数
- `observation`：工具返回的观察结果

```python
AgentResult
```

作用：

- 记录 Agent 的最终结果

字段：

- `final_answer`：最终答案
- `steps`：完整 ReAct 执行轨迹

### agent/prompts.py

作用：集中管理 Prompt 模板。

变量：

```python
SYSTEM_PROMPT
```

作用：

- 告诉大模型它是 LitCraft Agent
- 告诉大模型必须使用 ReAct 格式
- 约束大模型必须返回合法 JSON
- 说明工具调用格式和最终回答格式

函数：

```python
build_user_prompt(topic, tools, scratchpad) -> str
```

作用：

- 把用户研究主题、工具说明、历史执行步骤拼成一轮 user prompt
- 每一轮 ReAct 循环都会重新调用它

### agent/langgraph_agent.py

作用：实现 ReAct Agent 主循环（基于 LangGraph 框架）。

类：

```python
class LangGraphAgent:
```

作用：

- 使用 LangGraph 编排 Agent 状态机
- 控制 Agent 一轮一轮地思考（thought → action → observation）
- 解析大模型返回的 JSON 决策
- 根据 action 调用工具
- 把 observation 写回状态上下文
- 在模型返回 `final_answer` 时结束循环

方法：

```python
__init__(llm, tools, max_steps=15, temperature=0)
```

- `llm` — 大模型客户端
- `tools` — `ToolRegistry` 实例
- `max_steps` — 最大执行步数（默认 20）
- `temperature` — 模型采样温度

```python
run(topic) -> AgentResult
```

- 构造 AgentState 图
- 每轮构造 Prompt 并调用 LLM
- 判断是调用工具还是输出最终答案
- 返回 `AgentResult`（含 final_answer 和完整 steps）

```python
_build_scratchpad(steps) -> str
```

- 将历史执行步骤格式化为文本，供大模型参考

```python
_parse_decision(raw_response) -> dict
```

- 解析大模型返回的 JSON 字符串 → Python 字典
- 包含 JSON 修复逻辑：适配模型可能输出的不标准格式

### tools/base.py

作用：定义工具的基础协议。

类：

```python
ToolSpec
```

作用：

- 描述一个工具的名称、功能、输入格式
- 这个结构会被放进 Prompt，让大模型知道能调用什么工具

```python
Tool
```

作用：

- 所有工具的抽象父类
- 以后真实 arXiv 搜索工具、PDF 解析工具、RAG 检索工具都要继承它

方法：

```python
run(tool_input) -> str
```

作用：

- 执行工具
- 返回文本形式的 observation

```python
spec() -> ToolSpec
```

作用：

- 把工具信息转换成结构化说明
- 给 `ToolRegistry.render_descriptions()` 使用

### tools/registry.py

作用：管理所有工具。

类：

```python
ToolRegistry
```

作用：

- 注册工具
- 根据工具名查找工具
- 执行工具
- 把工具列表渲染成 Prompt 里的工具说明

方法：

```python
register(tool) -> None
```

作用：

- 把一个工具加入注册表

```python
run(name, tool_input) -> str
```

作用：

- 根据工具名找到工具
- 调用工具的 `run()`
- 返回 observation

```python
render_descriptions() -> str
```

作用：

- 把所有工具的 `name`、`description`、`input_schema` 转成 JSON 字符串
- 放进 Prompt 给大模型看

### tools/search/ 搜索工具

提供多个学术数据源的搜索能力。

#### arxiv_search.py

基于 arXiv API 的论文搜索。

```python
class ArxivSearchTool
```

- `run(tool_input)` — 根据 `query`、`limit`（可选）、`year_from`（可选）搜索 arXiv
- 返回论文列表（标题、作者、摘要、年份、arXiv ID、PDF 链接）

#### semantic_scholar.py

基于 Semantic Scholar API 的论文搜索。

```python
class SemanticScholarTool
```

- `run(tool_input)` — 根据 `query`、`limit` 搜索
- 内置指数退避重试机制应对 429 限流
- 返回论文标题、作者、年份、摘要、引用数、PDF 链接

#### google_scholar.py

基于 `scholarly` 库的 Google Scholar 搜索。

```python
class GoogleScholarTool
```

- `run(tool_input)` — 使用指定关键词搜索
- 解析 `Publication` 对象的 `bib` 字典获取元数据
- 包含代理配置支持（`GOOGLE_SCHOLAR_PROXY` 环境变量）

#### multi_source_search.py

一站式多源聚合搜索，整合上述三个来源的结果。

```python
class MultiSourceSearchTool
```

- `run(tool_input)` — 一个调用同时搜索 arXiv + Semantic Scholar + Google Scholar
- 即使部分来源失败，仍返回其他成功来源的结果
- 去重后合并返回

### tools/paper_downloader.py

下载论文 PDF。

```python
class PaperDownloaderTool
```

- `run(tool_input)` — 接收论文 URL 和标题，下载 PDF 到 `output/papers/`
- 支持 arXiv 和 Semantic Scholar PDF 链接

### tools/pdf_parser.py

解析 PDF 文件提取文本。

```python
class PDFParserTool
```

- `run(tool_input)` — 读取 PDF 路径，解析出纯文本内容

### tools/pdf_generator.py

将最终文献综述输出为 PDF 文件。

```python
class PDFGeneratorTool
```

- `run(tool_input)` — 接收 Markdown 综述文本，生成 PDF 保存到 `output/`

### tools/text_chunker.py

将长文本按指定大小分块。

```python
class TextChunkerTool
```

- `run(tool_input)` — 接收文本，返回分块列表（默认 512 字符 + 128 重叠）

### tools/vector_store.py

封装 Chroma 向量数据库操作。

```python
class VectorStoreTool
```

- `run(tool_input)` — 将文本块及其元数据批量存入 Chroma
- 使用 `all-MiniLM-L6-v2` 模型生成嵌入向量

### tools/advanced_retrieval.py

混合策略的智能检索工具。

```python
class AdvancedRetrievalTool
```

- `run(tool_input)` — 对查询执行 **HyDE**（假设文档嵌入）+ **MQE**（多查询扩展）双策略检索
- 合并两种检索结果，返回最相关的内容块

### api/server.py

作用：FastAPI 后端服务入口，提供 RESTful API 供前端调用。

```python
app = FastAPI()
```

API 路由：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agent/start` | 启动新综述任务（异步） |
| GET | `/api/agent/status/{task_id}` | 查询任务状态 |
| GET | `/api/agent/result/{task_id}` | 获取最终结果 |
| GET | `/api/agent/steps/{task_id}` | 获取执行步骤（可轮询） |
| GET | `/api/output/{filename}` | 下载生成的 PDF 文件 |
| GET | `/api/history` | 历史任务列表 |

### api/task_manager.py

作用：后台任务管理器，负责在独立线程中异步执行 Agent 任务。

```python
class TaskManager
```

- `start_task(topic, year_from, save_pdf)` — 在线程池中启动一个新任务，返回 task_id
- `get_status(task_id)` — 查询任务状态（running / done / error）
- `get_result(task_id)` — 获取任务最终结果
- `get_steps(task_id)` — 获取执行步骤列表（用于前端实时更新）
- `list_tasks()` — 获取所有历史任务摘要

核心设计：使用 `ThreadPoolExecutor(max_workers=2)` 避免阻塞 FastAPI 事件循环，任务状态存储在内存中。

### api/schemas.py

作用：定义 API 请求和响应的 Pydantic 数据结构，用于自动校验和 API 文档生成。

类：

```python
AgentStartRequest
```

- `topic` — 研究主题
- `year_from` — 年份过滤条件
- `save_pdf` — 是否自动生成 PDF

```python
AgentResultResponse
```

- `final_answer` — 最终综述正文（Markdown）
- `steps` — 完整 ReAct 执行轨迹列表
- `pdf_path` — PDF 文件保存路径

### frontend/app.py

作用：Streamlit 前端主页面，为用户提供可视化操作界面。

布局与功能：

| 页面 | 功能 |
|------|------|
| 📝 新建综述 | 输入研究主题、年份过滤、PDF 选项，提交任务 |
| 📊 执行状态 | 实时轮询展示每步思考→行动→观察，完成后展示综述正文和 PDF 下载 |
| 📚 历史记录 | 按创建时间倒序列出所有历史任务，可查看完成结果 |

通过 `requests` 库调用 FastAPI 后端接口，实现前后端分离。

## 主流程调用图

```text
用户运行 main.py --topic "..." [--year-from 2020] [--save-pdf output/review.pdf]
    │
    ▼
main.py 调用 build_agent()
    │  ├─ 创建 LitCraftAgentsLLM（读取 .env 配置）
    │  ├─ 创建 ToolRegistry
    │  │   ├─ 注册 MultiSourceSearchTool
    │  │   ├─ 注册 ArxivSearchTool
    │  │   ├─ 注册 SemanticScholarTool
    │  │   ├─ 注册 GoogleScholarTool
    │  │   ├─ 注册 PaperDownloaderTool
    │  │   ├─ 注册 PDFParserTool
    │  │   ├─ 注册 PDFGeneratorTool
    │  │   ├─ 注册 TextChunkerTool
    │  │   ├─ 注册 VectorStoreTool
    │  │   └─ 注册 AdvancedRetrievalTool
    │  └─ 创建 LangGraphAgent(llm, tools)
    │
    ▼
agent.run(topic) 进入 ReAct 循环
    │
    ├─1. 构造 Prompt = SYSTEM_PROMPT + build_user_prompt(topic, tools, scratchpad)
    ├─2. LLM.chat() → 模型输出 JSON
    ├─3. _parse_decision() → { "thought": ..., "action": ..., "action_input": {...} }
    │     或 { "thought": ..., "final_answer": ... }
    │
    ├─4. 如果有 action：
    │     ├─ ToolRegistry.run(action, action_input)
    │     ├─ 工具返回 observation
    │     ├─ 追加到 scratchpad
    │     └─ 回到步骤 1
    │
    └─5. 如果有 final_answer：
          ├─ 组装 AgentResult.final_answer
          └─ main.py 打印最终答案

    循环条件：steps < max_steps 且尚未收到 final_answer
```

## 配置参考

### `.env` 文件

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_MODEL_ID` | ✅ | 模型名（如 `gpt-4o`、`deepseek-chat`） |
| `LLM_API_KEY` | ✅ | API 密钥 |
| `LLM_BASE_URL` | ✅ | OpenAI-compatible API 地址 |
| `LLM_TIMEOUT` | ❌ | 请求超时（秒，默认 60） |
| `GOOGLE_SCHOLAR_PROXY` | ❌ | Google Scholar 代理地址（如 `http://127.0.0.1:10808`） |

### 环境要求

- Python 3.10+
- 依赖见 `requirements.txt`
- 推荐使用 Conda 管理环境
