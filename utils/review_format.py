"""把综述 Markdown 收成中文学术论文体例：编号标题、文末参考文献、去掉章末题录。"""

from __future__ import annotations

import re

_REF_HEAD = re.compile(r"(?im)^#{1,3}\s*(?:\d+(?:\.\d+)*\s+)?参考文献\b")
_INLINE_BIB_HEAD = re.compile(
    r"(?im)^#{0,3}\s*(依据题录摘要|本节引用|本章参考文献|引用文献|参考文献清单)\b"
)
_BIB_ENTRY = re.compile(r"^\[\d+\]\s+\S.{8,}$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# 章末题录：`[3] CW-DETR: ... (2026) | Wang T | [4] LRD-DETR: ...`
_DUMP_ENTRY = re.compile(
    r"(?:\s*\|\s*)*\[\d+\]\s+[A-Za-z].+?"
    r"(?=(?:\s*\|\s*)*\[\d+\]\s+[A-Za-z]|$)"
)

KNOWN_CHAPTERS = [
    "引言",
    "问题定义与任务设定",
    "方法与技术路线",
    "数据、评价与实验",
    "争议、不足与开放问题",
    "未来方向",
    "结论",
    "参考文献",
]


def repair_broken_identifiers(text: str) -> str:
    """把换行拆开的型号拼回去，例如 YOLOv1 1 -> YOLOv11。"""
    out = text or ""
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9+._/-]*\d)\s+(\d{1,3})(?=[^\dA-Za-z]|$)")
    prev = None
    while prev != out:
        prev = out
        out = pattern.sub(r"\1\2", out)
    return out


def document_title(topic: str) -> str:
    """综述题名：主题后补「综述」，已带则不再重复。"""
    text = (topic or "").strip()
    if not text:
        return "文献综述"
    if text.endswith("综述") or text.endswith("研究进展"):
        return text
    return f"{text}综述"


def numbered_heading(index: int, heading: str, level: int = 2) -> str:
    """`## 1 引言` 这种带章号的 Markdown 标题。"""
    prefix = "#" * max(1, min(int(level), 6))
    title = (heading or "").strip()
    return f"{prefix} {int(index)} {title}".rstrip()


def ensure_section_heading(markdown: str, index: int, heading: str) -> str:
    """保证一节以带编号的二级标题开头。"""
    wanted = numbered_heading(index, heading, level=2)
    text = (markdown or "").strip()
    if not text:
        return wanted
    first, sep, rest = text.partition("\n")
    if re.match(r"^#{1,3}\s+\S", first.strip()):
        return f"{wanted}\n{rest}" if sep else wanted
    return f"{wanted}\n\n{text}"


def ensure_document_title(text: str, topic: str = "") -> str:
    """文首补一级标题（题名），已有 `# 标题` 则不动。"""
    body = (text or "").lstrip()
    if re.match(r"^#\s+\S", body) and not body.startswith("##"):
        return body
    title = document_title(topic)
    if title and body.startswith(f"# {title}"):
        return body
    return f"# {title}\n\n{body}" if title else body


_INLINE_PHRASE = re.compile(r"本节依据题录摘要编写[。．]?")
_INLINE_LABEL = re.compile(r"依据题录摘要\s*(?:\[[0-9]+\]\s*)*[。．]?")


def strip_inline_bibliographies(text: str) -> str:
    """去掉各章下面的题录清单，只保留文末那一节参考文献。"""
    raw = text or ""
    matches = list(_REF_HEAD.finditer(raw))
    if matches:
        last = matches[-1]
        body, refs = raw[: last.start()], raw[last.start() :]
        if _has_later_body_heading(raw[last.end() :]):
            body, refs = raw, ""
        else:
            body = _drop_mid_document_ref_sections(body)
    else:
        body, refs = raw, ""
    body = _strip_bib_blocks(body)
    body = _strip_dumped_citations(body)
    body = _INLINE_PHRASE.sub("", body)
    body = _INLINE_LABEL.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"([。．])\s*([。．])", r"\1", body)
    refs = _normalize_reference_lines(refs.strip()) if refs.strip() else ""
    if refs:
        return f"{body}\n\n{refs}".strip() if body else refs
    return body


def polish_review_markdown(text: str, topic: str = "") -> str:
    """成稿后整理：去章末文献、补题名、压缩空行。"""
    cleaned = _untangle_headings(text or "")
    cleaned = strip_inline_bibliographies(cleaned)
    cleaned = _untangle_headings(cleaned)
    cleaned = ensure_document_title(cleaned, topic)
    cleaned = repair_broken_identifiers(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def clean_inline_markdown(text: str) -> str:
    """去掉粗体/斜体/行内代码/Markdown 链接，保留正文与 [n] 引用。"""
    s = (text or "").strip()
    s = _MD_LINK.sub(r"\1", s)
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"(?<!\*)\*(?!\*)", "", s)
    return s.strip()


def match_chapter_heading(line: str) -> tuple[int, str, str] | None:
    """识别 `引言` / `1 引言` / `## 1 引言`，正文若粘在标题后则拆开。"""
    s = (line or "").strip()
    if not s:
        return None
    s = re.sub(r"^#{1,3}\s*", "", s)
    for i, heading in enumerate(KNOWN_CHAPTERS, 1):
        numbered = re.match(rf"^{i}[.\s、]+{re.escape(heading)}(.*)$", s)
        if numbered:
            return i, heading, numbered.group(1).strip()
        if s == heading:
            return i, heading, ""
        if s.startswith(heading) and len(s) > len(heading):
            return i, heading, s[len(heading) :].strip()
    return None


def _chapter_name_pattern() -> str:
    return "|".join(re.escape(h) for h in KNOWN_CHAPTERS)


def _untangle_headings(text: str) -> str:
    """把粘在同一行的「## 1 引言正文」或「句末。## 2 标题」拆开。"""
    names = _chapter_name_pattern()
    text = text or ""
    text = re.sub(
        rf"(?m)^(#+[ \t]+\d+[ \t]+(?:{names}))(?=\S)",
        r"\1\n\n",
        text,
    )
    text = re.sub(
        rf"(?<=[\u4e00-\u9fffA-Za-z0-9。．！？；;）\]])"
        rf"(#+[ \t]+\d+[ \t]+(?:{names}))",
        r"\n\n\1",
        text,
    )
    return text


def split_reference_entries(text: str) -> list[str]:
    """把挤在一行里的 `[1] ... [2] ...` 拆成多条。"""
    blob = re.sub(r"\s+", " ", (text or "").strip())
    if not blob:
        return []
    parts = re.split(r"(?=\[\d+\])", blob)
    out: list[str] = []
    for part in parts:
        item = part.strip(" |")
        if not item:
            continue
        if re.match(r"^\[\d+\]", item):
            out.append(item)
        elif out:
            out[-1] = f"{out[-1]} {item}".strip()
        else:
            out.append(item)
    return out


def _normalize_reference_lines(refs: str) -> str:
    """参考文献一节：标题下一行一条文献。"""
    if not (refs or "").strip():
        return ""
    lines = refs.splitlines()
    heading = lines[0].strip() if lines else "## 参考文献"
    rest = " ".join(ln.strip() for ln in lines[1:] if ln.strip())
    entries = split_reference_entries(rest)
    if not entries:
        return heading
    return heading + "\n\n" + "\n".join(entries)


def _strip_dumped_citations(body: str) -> str:
    """去掉粘在段末的完整题录，保留正文里的 [1][2]。"""
    chunks = re.split(r"(?m)(^(?:#{1,3}[ \t].+)$)", body or "")
    out: list[str] = []
    for chunk in chunks:
        if re.match(r"(?m)^#{1,3}[ \t]", chunk or ""):
            out.append("\n\n" + chunk.strip() + "\n\n")
            continue
        paras = re.split(r"\n\s*\n", chunk)
        cleaned: list[str] = []
        for para in paras:
            text = para.strip()
            if not text:
                continue
            text = _DUMP_ENTRY.sub("", text)
            text = re.sub(r"(?:\s*\|\s*)+$", "", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            if text:
                cleaned.append(text)
        out.append("\n\n".join(cleaned))
    return "".join(out)


def _has_later_body_heading(after: str) -> bool:
    for line in (after or "").splitlines():
        s = line.strip()
        if re.match(r"^#{1,3}\s+\S", s) and not _REF_HEAD.match(s):
            return True
    return False


def _drop_mid_document_ref_sections(body: str) -> str:
    """丢掉误插在正文中间的「参考文献」小段，直到下一个真正标题。"""
    out: list[str] = []
    skipping = False
    for line in (body or "").splitlines():
        s = line.strip()
        if _REF_HEAD.match(s):
            skipping = True
            continue
        if skipping:
            if re.match(r"^#{1,3}\s+\S", s) and not _REF_HEAD.match(s):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def _strip_bib_blocks(body: str) -> str:
    """去掉「依据题录摘要」及其后的 [n] 题录行。"""
    out: list[str] = []
    skipping = False
    for line in (body or "").splitlines():
        s = line.strip()
        if _INLINE_BIB_HEAD.match(s) or s.startswith("依据题录摘要"):
            skipping = True
            continue
        if skipping:
            if not s or _BIB_ENTRY.match(s):
                continue
            skipping = False
        if _BIB_ENTRY.match(s) and (len(s) > 40 or re.search(r"\b(19|20)\d{2}\b", s)):
            continue
        out.append(line)
    return "\n".join(out)
