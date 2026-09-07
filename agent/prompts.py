# 作用：集中管理 Agent 的提示词模板，并负责把主题、工具说明、历史步骤拼成 Prompt。
import re
from tools.registry import ToolRegistry

# 有可用文献时一律不少于此字数，不随篇数增减
REVIEW_MIN_CHARS = 3000

REVIEW_WRITING_SPEC = """# 综述体例（必须遵守）

这是一篇中文学术文献综述，不是摘要、不是检索报告、不是「可以开始写了」的状态句。

## 结构（用 Markdown 二级标题，按此顺序，可在节内再分小节）

1. 引言：研究背景、问题为什么重要、综述范围与组织方式
2. 问题定义与任务设定：术语、输入输出、与相邻任务的区别
3. 方法与技术路线：按技术路线分类评述（不要写成「第一篇、第二篇」流水账），比较思路、适用场景与局限
4. 数据、评价与实验：常用数据集、指标、代表性结果（有数字就写数字，没有就写「原文未给出」）
5. 争议、不足与开放问题
6. 未来方向
7. 结论：收束全文，不要只重复某一篇摘要
8. 参考文献：与正文 `[1]` `[2]` 一一对应

## 篇幅

- 只要检索到可用文献，正文不少于 3000 汉字，不随篇数加减
- 文献少：把已有工作写深，并说明覆盖范围有限；禁止编造未检索到的论文来凑篇幅
- 0 篇：不要凑字数，明确写未检索到可用文献，不要编造论文

## 证据与引用

- 只引用本次工具 Observation 中出现过的论文
- 有几篇可用文献就评述几篇，禁止只写其中一篇就交卷
- 没有全文时可根据题录/摘要写，并注明依据摘要
- 禁止用训练数据冒充本次检索；禁止把无关论文硬套到主题上
"""


def _strip_review_refs(text: str) -> str:
    cut = re.search(r"(?im)^#{1,3}\s*参考文献\b", text or "")
    if cut:
        return text[: cut.start()]
    return text or ""


def review_meets_standard(text: str, paper_count: int) -> bool:
    """判断交卷是否达到综述体例与最低字数。无文献时允许短说明。"""
    body = (text or "").strip()
    if paper_count <= 0:
        return bool(body) and (
            "未检索" in body or "没有找到" in body or "未找到" in body or "无可用" in body
        )

    lowered = body.lower()
    stub_marks = (
        "writing can begin",
        "can begin",
        "ready to write",
        "可以开始写",
        "可以动笔",
        "开始撰写",
        "开始写作",
    )
    if any(m in lowered or m in body for m in stub_marks):
        if len(body) < 800:
            return False

    min_chars = REVIEW_MIN_CHARS
    if len(body) < min_chars:
        return False

    headings = re.findall(r"(?m)^#{1,3}\s+\S+", body)
    if len(headings) < 4:
        return False
    if not re.search(r"(?im)^#{1,3}\s*参考文献\b|参考文献", body):
        return False
    if len(_strip_review_refs(body).strip()) < min_chars * 0.7:
        return False
    return True


SYSTEM_PROMPT = """你是 LitCraft Agent，一个用 ReAct（Reason + Act）完成文献综述的研究助手。

# 循环

每一步你先思考，再二选一：
1. 调用一个工具，等待环境返回 Observation，再根据观察决定下一步；
2. 认为证据已经足够（或确认无法获得更多证据）时，输出 `final_answer` 结束。

Observation 只来自工具真实返回，你不能编造观察。下一步必须依据上一步观察，而不是执行一张固定流水线。

# 任务

围绕用户研究主题，检索学术文献、按需获取全文、整合证据，写出可追溯的中文文献综述。
你不是在做闲聊，也不是用参数记忆写通识介绍。

# 如何根据观察决策（这才是 ReAct）

- 一次 `multi_source_search` = 同时查所有学术源，每源最多约 per_source_limit 篇候选，合并去重后保留约 final_limit 篇（默认 10），不是只搜两三篇。
- 搜索 0 篇：必须换 query 再搜（中英、同义、更宽/更窄）。
- 已有文献但覆盖不足：允许用**不同** query 再搜（例如中文主题补英文、补一个子方向）。同一 query 不要重复。
- 研究记忆里已列出的题录始终有效，不要因为近期 Observation 被压缩就以为没搜到。
- 搜到题录但没有 PDF：用记忆/观察里的 pdf_url 去下载，或用摘要写并注明无全文。
- 已入库（indexed_chunks>0）：用 `advanced_search` 取证据片段，或 `final_answer` 结束检索。
- 证据足够：输出 `final_answer`。不要在 JSON 里写整篇综述（系统会按研究记忆成稿，不少于 3000 字）。
- 多次换词仍无文献：`final_answer` 写明未检索到，不要编造。

允许、也鼓励你调整检索词和策略。用户主题是目标，不是唯一合法 query。

完整综述的结构与 3000 字底线由检索结束后的成稿步骤负责，本阶段不要把整篇综述塞进 JSON。

# 输出协议

每一步只返回一个 JSON 对象。第一个字符必须是 `{`，最后一个是 `}`。
禁止 markdown 代码块，禁止向用户提问，禁止写「下一步指示」「请告知您下一步」。你没有人类对话轮次，只能调用工具或 final_answer。

调用工具：
{"thought": "基于观察，下一步为什么这样做", "action": "精确工具名", "action_input": {"参数": "值"}}

结束检索（不要在 JSON 里塞整篇综述）：
{"thought": "为什么可以结束", "final_answer": "检索结束，请按体例成稿"}

规则：
- 不要输出 observation 字段。
- action 必须是本轮提供的精确工具名。
- final_answer 里的英文双引号请改成书名号《》或去掉，以免 JSON 崩掉。
- JSON 里的 final_answer 只需收束检索，完整综述由成稿步骤生成。
"""


JSON_REPAIR_PROMPT = """你刚才的输出不是合法 JSON，也不是对本系统的合法回复。
不要向用户提问，不要写「下一步指示」「请告知您下一步」。
只重新输出一个 JSON 对象，第一个字符必须是 { 。

调用工具：{"thought":"...","action":"工具名","action_input":{}}
结束检索：{"thought":"...","final_answer":"检索结束，请按体例成稿"}

上次输出开头：
"""


def build_user_prompt(
    topic: str,
    tools: ToolRegistry,
    scratchpad: str,
    year_from: str = "",
    per_source_limit: int = 10,
    final_limit: int = 10,
    memory_block: str = "",
) -> str:
    extras = []
    if year_from:
        extras.append(f"- 用户希望优先考虑 {year_from} 年及之后的文献；搜索时 year_from 必须用这个值。")
    else:
        extras.append("- 用户未限制发表年份：搜索时不要传 year_from / year，不要自行假设 2010 或近十年。")
    extras.append(
        f"- 数量参考：一次多源搜索各源最多约 {per_source_limit} 篇候选，合并去重后保留约 {final_limit} 篇。"
        "覆盖不足时换 query 再搜，新文献会并入研究记忆。"
    )
    extra_block = "\n".join(extras)
    memory_section = memory_block.strip() or "（研究记忆为空）"

    return f"""# 本轮任务

研究主题：{topic}

{extra_block}

{memory_section}

可用工具（action 必须用下面的精确名称）：
{tools.render_descriptions()}

# 近期工作记忆（Thought / Action / Observation，旧步骤会压缩）

{scratchpad if scratchpad else "（还没有步骤。请先检索。）"}

# 现在

先看研究记忆里已有哪些文献和检索词。同一 query 不要再搜；需要补全再用不同 query。
可以 advanced_search 或 final_answer。
第一个字符必须是 {{ ，只输出一个 JSON 对象，不要向用户提问。"""
