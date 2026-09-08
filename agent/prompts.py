# 作用：集中管理 Agent 的提示词。本项目固定两套主 Prompt + 一个纠错附录。
# 1) 决策套 REACT_*：只负责调用工具（原生 function calling，失败则 JSON）
# 2) 成稿套 WRITER_*：按节 retrieve-and-write，不调工具
# 记忆 / 题录 / 检索片段由函数现拼进 User，不单独做成第三套 System。
from __future__ import annotations

import re
from typing import Any

from tools.registry import ToolRegistry

# 有可用文献时一律不少于此字数，不随篇数增减
REVIEW_MIN_CHARS = 3000

REVIEW_WRITING_SPEC = """# 综述体例（必须遵守）

这是一篇中文学术文献综述，不是摘要、不是检索报告、不是「可以开始写了」的状态句。

## 结构（一级标题必须带序号，按此顺序，可在节内再分 `### 1.1` 小节）

`## 1 引言`：研究背景、问题为什么重要、综述范围与组织方式
`## 2 问题定义与任务设定`：术语、输入输出、与相邻任务的区别
`## 3 方法与技术路线`：按技术路线分类评述（不要写成「第一篇、第二篇」流水账），比较思路、适用场景与局限
`## 4 数据、评价与实验`：常用数据集、指标、代表性结果（有数字就写数字，没有就写「原文未给出」）
`## 5 争议、不足与开放问题`
`## 6 未来方向`
`## 7 结论`：收束全文，不要只重复某一篇摘要
`## 8 参考文献`：由系统生成；你不要写这一节

## 篇幅

- 只要检索到可用文献，正文不少于 3000 汉字，不随篇数加减
- 文献少：把已有工作写深，并说明覆盖范围有限；禁止编造未检索到的论文来凑篇幅
- 0 篇：不要凑字数，明确写未检索到可用文献，不要编造论文

## 证据与引用

- 只引用题录白名单中的论文；有原文片段时优先依据片段，不要只改写摘要
- 有几篇可用文献就评述几篇，禁止只写其中一篇就交卷
- 正文用 `[1][2]` 标注出处；不要在各章末列出文献题录，不要写「依据题录摘要」小节
- 没有全文时可根据题录/摘要写正文，同样只用 `[n]` 引用
- 参考文献只出现在全文最后一节，与正文编号一一对应
- 禁止用训练数据冒充本次检索；禁止把无关论文硬套到主题上
"""


def _strip_review_refs(text: str) -> str:
    cut = re.search(r"(?im)^#{1,3}\s*(?:\d+(?:\.\d+)*\s+)?参考文献\b", text or "")
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
    if not re.search(r"(?im)^#{1,3}\s*(?:\d+(?:\.\d+)*\s+)?参考文献\b|参考文献", body):
        return False
    if len(_strip_review_refs(body).strip()) < min_chars * 0.7:
        return False
    return True


REACT_SYSTEM_NATIVE = """你是 LitCraft Agent，用工具完成文献检索，不在这一步写综述。

每一步调用一个工具，或调用 final_answer 结束检索。
不要向用户提问，不要写「下一步指示」。完整综述由系统在检索结束后按体例生成。

- 第一次搜索：query 必须等于用户研究主题全文，禁止改写成单个词（例如不要搜「循环」「天气」这种残词）。
- 一次 multi_source_search 会查所有学术源并尽量下载入库，不是只搜两三篇。
- 同一 query 不要重复；覆盖不足才换不同 query（中英、子方向），新词仍须围绕原主题。
- 不要自己填 year_from、limit、per_source_limit，这些由系统写入。
- 证据足够或多次换词仍无文献：调用 final_answer。
"""


REACT_SYSTEM = """你是 LitCraft Agent，一个用 ReAct（Reason + Act）完成文献综述的研究助手。

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

SYSTEM_PROMPT = REACT_SYSTEM  # 兼容旧引用


JSON_REPAIR_PROMPT = """你刚才的输出不是合法 JSON，也不是对本系统的合法回复。
不要向用户提问，不要写「下一步指示」「请告知您下一步」。
只重新输出一个 JSON 对象，第一个字符必须是 { 。

调用工具：{"thought":"...","action":"工具名","action_input":{}}
结束检索：{"thought":"...","final_answer":"检索结束，请按体例成稿"}

上次输出开头：
"""


WRITER_SYSTEM = """你是学术文献综述作者。只根据用户提供的题录白名单和原文片段写作；必须满足给定体例和字数。不要输出 JSON，不要用代码围栏包裹全文。"""

WRITER_SECTION_SYSTEM = """你是学术文献综述作者。只写用户指定的那一节 Markdown。
必须承接前文已出现的术语、问题定义和技术分类，不要把引言背景再写一遍，不要预告后面章节的内容。
只引用题录白名单中的编号，正文用 [1][2] 标注即可。
禁止在节末罗列文献题录，禁止写「依据题录摘要」「本节引用」或任何参考文献列表；参考文献由系统在全文最后统一生成。
不要写其他节。不要 JSON，不要代码围栏。"""

# 按节检索：方法节搜方法、实验节搜数据
REVIEW_SECTIONS: list[dict[str, Any]] = [
    {
        "heading": "引言",
        "query": "background motivation importance problem",
        "target_chars": 400,
    },
    {
        "heading": "问题定义与任务设定",
        "query": "task definition input output terminology",
        "target_chars": 350,
    },
    {
        "heading": "方法与技术路线",
        "query": "method architecture algorithm network model CNN transformer training",
        "target_chars": 750,
    },
    {
        "heading": "数据、评价与实验",
        "query": "dataset benchmark metric experiment result accuracy precision recall",
        "target_chars": 650,
    },
    {
        "heading": "争议、不足与开放问题",
        "query": "limitation challenge open problem weakness",
        "target_chars": 350,
    },
    {
        "heading": "未来方向",
        "query": "future direction outlook trend",
        "target_chars": 300,
    },
    {
        "heading": "结论",
        "query": "conclusion summary contribution",
        "target_chars": 300,
    },
]


def section_search_query(topic: str, section: dict[str, Any]) -> str:
    """成稿某一节用的检索 query：主题 + 该节关键词。"""
    extra = str(section.get("query") or "").strip()
    topic = (topic or "").strip()
    return f"{topic} {extra}".strip()


def build_react_user(
    topic: str,
    tools: ToolRegistry,
    scratchpad: str,
    year_from: str = "",
    per_source_limit: int = 10,
    final_limit: int = 10,
    memory_block: str = "",
    native_tools: bool = False,
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
    if native_tools:
        closing = (
            "先看研究记忆里已有哪些文献和检索词。第一次搜索必须用上面的研究主题原文当 query。\n"
            "请调用一个函数；检索结束时调用 final_answer。不要自己填 limit / per_source_limit。"
        )
    else:
        closing = (
            "先看研究记忆里已有哪些文献和检索词。同一 query 不要再搜；需要补全再用不同 query。\n"
            "可以 advanced_search 或 final_answer。\n"
            "第一个字符必须是 { ，只输出一个 JSON 对象，不要向用户提问。"
        )

    return f"""# 本轮任务

研究主题：{topic}

{extra_block}

{memory_section}

可用工具（action 必须用下面的精确名称）：
{tools.render_descriptions()}

# 近期工作记忆（Thought / Action / Observation，旧步骤会压缩）

{scratchpad if scratchpad else "（还没有步骤。请先检索。）"}

# 现在

{closing}"""


build_user_prompt = build_react_user  # 兼容旧引用


def _format_authors(authors: Any) -> str:
    if isinstance(authors, list):
        return ", ".join(str(a) for a in authors[:8])
    return str(authors or "")


def format_paper_whitelist(topic: str, papers: list[dict[str, Any]]) -> str:
    lines = [f"研究主题：{topic}", "", "题录白名单（只能引用这些，禁止编造）："]
    if not papers:
        lines.append("（没有题录。请明确写未检索到可用文献，不要编造。）")
        return "\n".join(lines)
    for i, paper in enumerate(papers, 1):
        title = paper.get("title") or paper.get("paper_title") or ""
        year = paper.get("year") or ""
        authors = _format_authors(paper.get("authors"))
        venue = paper.get("venue") or paper.get("journal") or ""
        abstract = str(paper.get("abstract") or paper.get("snippet") or paper.get("text") or "")[:800]
        url = paper.get("url") or paper.get("pdf_url") or ""
        lines.append(
            f"[{i}] {title} ({year}) | {authors} | {venue}\n"
            f"    链接: {url}\n"
            f"    摘要: {abstract or '（观察中无摘要）'}"
        )
    return "\n".join(lines)


def format_retrieved_passages(passages: list[dict[str, Any]] | None) -> str:
    if not passages:
        return "（没有检索到原文片段。请主要依据上面的摘要撰写正文，用 [n] 引用即可，不要另列文献题录。）"
    lines = ["检索原文片段（优先依据这些段落写作，引用序号必须对上题录白名单）："]
    for i, item in enumerate(passages, 1):
        title = str(item.get("title") or item.get("source") or "").strip() or "（未知来源）"
        cite = item.get("cite") or ""
        prefix = f"[{cite}] " if cite else ""
        text = str(item.get("content") or "").strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "…"
        lines.append(f"{i}. {prefix}{title}\n    {text or '（空）'}")
    return "\n".join(lines)


def build_writer_user(
    topic: str,
    papers: list[dict[str, Any]],
    passages: list[dict[str, Any]] | None = None,
) -> str:
    """成稿 User：白名单 + 检索片段 + 体例。"""
    whitelist = format_paper_whitelist(topic, papers)
    evidence = format_retrieved_passages(passages)
    body = (
        f"{whitelist}\n\n"
        f"{evidence}\n\n"
        f"{REVIEW_WRITING_SPEC}\n"
        "请直接输出完整 Markdown 综述，不要 JSON，不要代码围栏。"
    )
    if len(body) > 24000:
        body = body[:24000] + "\n…（证据已截断）"
    return body


def section_number(section: dict[str, Any]) -> int:
    """当前节在 REVIEW_SECTIONS 中的序号，从 1 起。"""
    heading = str(section.get("heading") or "")
    for i, spec in enumerate(REVIEW_SECTIONS, 1):
        if spec.get("heading") == heading:
            return i
    return 1


def section_heading_md(section: dict[str, Any]) -> str:
    """写作时使用的二级标题，例如 `## 1 引言`。"""
    heading = str(section.get("heading") or "本节")
    return f"## {section_number(section)} {heading}"


def format_references(papers: list[dict[str, Any]]) -> str:
    lines = [f"## {len(REVIEW_SECTIONS) + 1} 参考文献", ""]
    if not papers:
        lines.append("（无）")
        return "\n".join(lines)
    for i, paper in enumerate(papers, 1):
        title = paper.get("title") or paper.get("paper_title") or "（无标题）"
        year = paper.get("year") or ""
        authors = _format_authors(paper.get("authors"))
        url = paper.get("url") or paper.get("pdf_url") or ""
        bit = f"[{i}] {title}"
        if authors:
            bit += f". {authors}"
        if year:
            bit += f". {year}"
        if url:
            bit += f". {url}"
        lines.append(bit)
    return "\n".join(lines)


def review_outline() -> str:
    lines = ["全文结构（按此顺序，你只写当前指定的那一节）："]
    for i, spec in enumerate(REVIEW_SECTIONS, 1):
        lines.append(f"{i}. {spec.get('heading')}")
    lines.append(f"{len(REVIEW_SECTIONS) + 1}. 参考文献（系统生成，你不要写）")
    return "\n".join(lines)


def compact_prior_sections(parts: list[str], max_each: int = 500, max_total: int = 3600) -> str:
    """把已写成的节压短，供后文承接，避免每节像独立短文。"""
    if not parts:
        return "（这是第一节，开篇即可。）"
    blobs: list[str] = []
    total = 0
    for block in parts:
        text = " ".join((block or "").split())
        if len(text) > max_each:
            text = text[:max_each] + "…"
        blobs.append(text)
        total += len(text)
        if total >= max_total:
            break
    return "已写章节（请承接，不要重复）：\n" + "\n".join(blobs)


def build_section_writer_user(
    topic: str,
    section: dict[str, Any],
    papers: list[dict[str, Any]],
    passages: list[dict[str, Any]] | None = None,
    prior_sections: list[str] | None = None,
) -> str:
    target = int(section.get("target_chars") or 400)
    whitelist = format_paper_whitelist(topic, papers)
    evidence = format_retrieved_passages(passages)
    missing = ""
    if not passages:
        missing = (
            "本节没有检索到专属原文片段。不要跳过本节，请依据题录摘要写正文；"
            "用 [n] 引用即可，不要另起「依据题录摘要」小节，不要罗列文献题录。"
        )
    heading_md = section_heading_md(section)
    body = (
        f"{review_outline()}\n\n"
        f"{compact_prior_sections(list(prior_sections or []))}\n\n"
        f"{whitelist}\n\n"
        f"{evidence}\n\n"
        f"{missing}\n"
        f"只写这一节，二级标题必须写成：{heading_md}\n"
        f"目标约 {target} 字。承接上文术语与分类，用 [1][2] 引用。\n"
        "不要写其他节，不要写参考文献，不要把前面章节重写一遍。"
    )
    if len(body) > 18000:
        body = body[:18000] + "\n…（证据已截断）"
    return body
