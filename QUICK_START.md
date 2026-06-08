# 🚀 LitCraft Agent 快速开始

## 项目完整性检查 ✅

### 6 个已集成工具
| 序号 | 工具名 | 文件 | 功能 | 状态 |
|-----|------|------|------|-----|
| 1️⃣ | arxiv_search | `tools/arxiv_search.py` | 搜索学术论文 | ✅ |
| 2️⃣ | paper_downloader | `tools/paper_downloader.py` | 下载 PDF 文件 | ✅ |
| 3️⃣ | pdf_parser | `tools/pdf_parser.py` | 解析 PDF 内容 | ✅ |
| 4️⃣ | text_chunker | `tools/text_chunker.py` | 智能文本分块 | ✅ |
| 5️⃣ | vector_store | `tools/vector_store.py` | 向量数据库 | ✅ |
| 6️⃣ | advanced_search | `tools/advanced_retrieval.py` | HyDE+MQE 混合检索 | ✅ |

### Agent 框架
- **框架**：LangGraph StateGraph ✅
- **LLM**：DeepSeek (OpenAI 兼容) ✅
- **向量库**：Chroma + sentence-transformers ✅
- **工具系统**：Tool ABC + ToolRegistry ✅

### 编译状态
- **错误数**：0 ✅
- **导入状态**：全部成功 ✅
- **文件结构**：清洁有序 ✅

---

## 使用方式

### 1. 环境配置
```bash
# 激活 conda 环境
conda activate litcraft

# 或者直接安装依赖
pip install -r requirements.txt
```

### 2. 设置 API 密钥（.env 或环境变量）
```
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL_ID=ep-20260602144126-qvm5j
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

### 3. 运行完整工作流
```bash
# 执行文献综述
python main.py --topic "Transformer 在自然语言处理中的应用"
```

### 4. 验证集成（可选）
```bash
# 检查所有工具是否正确注册
python verify_integration.py

# 演示高级检索策略
python demo_advanced_retrieval.py
```

---

## 工作流执行顺序

```
输入研究主题
    ↓
┌─────────────────────────────────────┐
│ 1️⃣ 搜索论文（arxiv_search）         │
│    输入：query                      │
│    输出：论文列表（id, title, ...） │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2️⃣ 下载 PDF（paper_downloader）     │
│    输入：url                        │
│    输出：已保存文件路径              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3️⃣ 解析内容（pdf_parser）           │
│    输入：pdf_path                   │
│    输出：提取的文本 + 元数据         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4️⃣ 分块文本（text_chunker）         │
│    输入：text                       │
│    输出：块列表（512 字符 + 128 重叠）│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5️⃣ 向量存储（vector_store）         │
│    输入：chunks                     │
│    输出：存储确认                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 6️⃣ 智能检索（advanced_search）      │
│    策略：HyDE + MQE 混合            │
│    输入：query                      │
│    输出：最相关的文本块              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 7️⃣ 生成综述（LLM 总结）             │
│    输入：retrieved documents        │
│    输出：文献综述                    │
└─────────────────────────────────────┘
    ↓
最终输出：格式化的文献综述
```

---

## 技术亮点

### ✨ 高级检索策略
- **HyDE（假设文档嵌入）**：LLM 生成 3 个假设文档，提高语义匹配
- **MQE（多查询扩展）**：LLM 生成 3 个查询变体，提高召回率
- **混合策略**：同时使用两者，融合优势 ⭐

### 🏗️ 工程化设计
- **指数退避重试**：自动处理 API 超时和网络错误
- **离线优先**：模型缓存到本地，支持离线运行
- **错误降级**：任何工具失败都有优雅降级方案
- **完整日志**：每一步都有详细输出

### 🔍 产品级质量
- LangGraph 行业标准框架
- 自定义 Tool 系统 + Registry 模式
- 完整的 RAG 管道实现
- 单元测试覆盖（test_*.py）

---

## 常见问题

### Q: 如何修改搜索结果数量？
```python
# 在 main.py build_agent() 中修改
tools.register(ArxivSearchTool(max_results=10))  # 改为 10 篇
```

### Q: 如何切换检索策略？
检索策略由 Agent 自动选择，或在 prompt 中明确指定：
```python
# 在 agent/prompts.py 中添加建议
"建议使用 strategy='hybrid' 获得最优结果"
```

### Q: 如何离线运行？
所有模型已缓存到 `./models`，只需配置 `LLM_*` 环境变量即可离线运行。

### Q: 如何扩展新工具？
1. 在 `tools/` 中创建新的 Tool 类（继承 `base.py` 中的 `Tool`）
2. 在 `main.py build_agent()` 中注册
3. 更新 `PROGRESS.md` 文档

---

## 项目文档

| 文档 | 用途 |
|-----|------|
| PROGRESS.md | 项目进度与完成度 |
| ADVANCED_RETRIEVAL.md | 高级检索详细说明 |
| README.md | 基础项目说明 |
| QUICK_START.md | 本文件，快速入门 |

---

## 下一步计划

- [ ] 集成 Streamlit 前端（用户界面）
- [ ] 实现论文元数据数据库（JSON/SQLite）
- [ ] 添加文献综述生成服务（跨论文聚合）
- [ ] FastAPI 后端（REST API）
- [ ] LangSmith 集成（Agent 调试）

---

**最后更新**：2024-06-07  
**项目状态**：✅ 功能完整，可投入使用  
**推荐用途**：面试作品集 / 技术博客 / 开源贡献
