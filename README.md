# LitCraft Agent

LitCraft Agent 是一个从 0 开始实现的智能文献综述 Agent 项目。

当前阶段目标：先跑通 Agent 内核，不急着做前端、后端、arXiv、PDF、RAG。

已经完成：

- LLM 调用封装
- LangGraph Agent 实现
- 工具调用协议
- Prompt 模板
- 命令行运行入口
- 完整 RAG 管道（arXiv 搜索、PDF 下载、PDF 解析、文本分块、向量存储、语义检索）

## 当前运行方式

先确认 `.env` 里有：

```env
LLM_MODEL_ID=你的模型名
LLM_API_KEY=你的 API Key
LLM_BASE_URL=你的 OpenAI-compatible 接口地址
LLM_TIMEOUT=60
```

然后运行：

```bash
pip install -r requirements.txt
python main.py --topic "RAG在医学问答中的应用"
```

如果你使用指定 Conda 环境，可以这样运行：

```bash
d:\Users\vv\Anaconda3\envs\litcraft\python.exe main.py --topic "你的研究主题"
```

## 当前目录结构

```text
LitCraft_Agent/
  main.py
  llm_client.py
  requirements.txt
  test_deepseek.py
  README.md

  agent/
    __init__.py
    models.py
    prompts.py
    react_agent.py

  tools/
    __init__.py
    base.py
    registry.py
    sample_tools.py
```

## 脚本说明

### main.py

作用：项目当前的命令行入口。

它负责：

- 读取用户传入的研究主题
- 创建大模型客户端
- 注册工具
- 创建 LangGraphAgent
- 运行 Agent
- 打印每一步思考、行动、观察和最终答案

函数：

```python
build_agent() -> LangGraphAgent
```

作用：

- 创建 `LitCraftAgentsLLM`
- 创建 `ToolRegistry`
- 注册多个工具（arXiv 搜索、PDF 下载、PDF 解析等）
- 返回组装好的 `LangGraphAgent`

```python
main() -> None
```

作用：

- 使用 `argparse` 读取命令行参数 `--topic`
- 调用 `build_agent()`
- 执行 `agent.run(topic)`
- 打印 Agent 执行轨迹和最终答案

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

### agent/react_agent.py

作用：实现 ReAct Agent 主循环。

类：

```python
class LangGraphAgent:
```

作用：

- 控制 Agent 一轮一轮地思考
- 解析大模型返回的 JSON 决策
- 根据 action 调用工具
- 把 observation 放回上下文
- 在模型返回 final_answer 时结束

方法：

```python
__init__(llm, tools, max_steps=6, temperature=0)
```

作用：

- 接收大模型客户端
- 接收工具注册表
- 设置最大执行步数
- 设置模型温度

```python
run(topic) -> AgentResult
```

作用：

- 执行完整 ReAct 循环
- 每轮构造 Prompt
- 调用 LLM
- 判断是调用工具还是最终回答
- 返回 `AgentResult`

```python
_build_scratchpad(steps) -> str
```

作用：

- 把历史步骤整理成文本
- 让大模型知道之前做过什么、工具返回过什么

```python
_parse_decision(raw_response) -> dict
```

作用：

- 把大模型返回的 JSON 字符串解析成 Python 字典
- 如果模型没有返回合法 JSON，会抛出错误

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

### tools/sample_tools.py

作用：提供第一阶段测试用的示例工具。

注意：这些还不是真实 arXiv 工具，只是为了先验证 ReAct 主循环能跑通。

类：

```python
KeywordPlannerTool
```

作用：

- 根据研究主题生成检索关键词方向

方法：

```python
run(tool_input) -> str
```

作用：

- 从 `tool_input` 中读取 `topic`
- 返回关键词规划结果

```python
MockPaperSearchTool
```

作用：

- 模拟论文搜索
- 后面会被真实 `ArxivSearchTool` 替换

方法：

```python
run(tool_input) -> str
```

作用：

- 从 `tool_input` 中读取 `query` 和 `limit`
- 返回模拟论文标题、年份、摘要

### test_deepseek.py

作用：你之前创建的大模型连接测试脚本。

它用于：

- 测试 `.env` 中的模型配置是否正确
- 直接调用一次 DeepSeek/OpenAI-compatible API
- 打印模型回复

这个脚本不参与当前 ReAct Agent 主流程。

## 脚本之间的调用逻辑

当前主流程如下：

```text
用户运行 main.py
    ↓
main.py 调用 build_agent()
    ↓
build_agent() 创建 LitCraftAgentsLLM
    ↓
build_agent() 创建 ToolRegistry
    ↓
build_agent() 注册 KeywordPlannerTool 和 MockPaperSearchTool
    ↓
build_agent() 创建 LangGraphAgent
    ↓
main.py 调用 agent.run(topic)
    ↓
LangGraphAgent.run() 调用 build_user_prompt()
    ↓
build_user_prompt() 读取 ToolRegistry.render_descriptions()
    ↓
LangGraphAgent.run() 调用 LitCraftAgentsLLM.chat()
    ↓
LitCraftAgentsLLM.chat() 调用 LitCraftAgentsLLM.think()
    ↓
think() 请求大模型
    ↓
大模型返回 JSON
    ↓
LangGraphAgent._parse_decision() 解析 JSON
    ↓
如果 JSON 中有 action：
    ↓
LangGraphAgent 调用 ToolRegistry.run(action, action_input)
    ↓
ToolRegistry 找到对应 Tool
    ↓
Tool.run() 返回 observation
    ↓
LangGraphAgent 把 observation 加入 steps
    ↓
进入下一轮 ReAct

如果 JSON 中有 final_answer：
    ↓
LangGraphAgent 返回 AgentResult
    ↓
main.py 打印最终答案
```

## 当前 ReAct 数据流

```text
topic
  ↓
Prompt = SYSTEM_PROMPT + 用户主题 + 工具说明 + 历史步骤
  ↓
LLM 输出 JSON 决策
  ↓
action / final_answer
  ↓
如果 action：执行工具，得到 observation
  ↓
observation 进入 scratchpad
  ↓
下一轮 Prompt
```

## 下一步计划

下一步建议实现真实工具：

```text
tools/arxiv_search.py
```

它会替换当前的 `MockPaperSearchTool`，让 Agent 真的去 arXiv 搜论文。

实现新脚本时，README 也会同步补充：

- 脚本作用
- 类和函数说明
- 输入输出格式
- 它和其他脚本的调用关系
