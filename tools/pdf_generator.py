"""将文献综述 Markdown 排成中文学术论文体例的 PDF（GB/T 7713 常见版式）。"""

from __future__ import annotations

import os
import re
from typing import Any

from tools.base import Tool
from utils.review_format import (
    clean_inline_markdown,
    polish_review_markdown,
    repair_broken_identifiers,
    split_reference_entries,
)

# A4 页边距约 2.54 cm；正文小四宋体、标题黑体、首行缩进 2 字符
_MARGIN_MM = 25.4
_TITLE_PT = 16       # 小三 黑体 居中
_H1_PT = 14          # 四号 黑体
_H2_PT = 12          # 小四 黑体
_BODY_PT = 12        # 小四 宋体
_REF_PT = 10.5       # 五号 宋体
_FOOT_PT = 9         # 小五
_BODY_H_MM = 7.4     # 约 1.5 倍行距
_REF_H_MM = 6.2
_INDENT_MM = 8.5     # 约 2 个汉字宽
_HEI_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]
_SONG_CANDIDATES = [
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simsun.ttf",
    "C:/Windows/Fonts/simfang.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
_TIMES_CANDIDATES = [
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/Times.ttf",
    "C:/Windows/Fonts/timesnr.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]
_FLOW_TOKEN = re.compile(
    r"(\[\d+\]|https?://[^\s]+|[A-Za-z][A-Za-z0-9._+/-]*|[0-9]+(?:\.[0-9]+)*|.)"
)
# 行首禁则：这些符号不能出现在一行开头
_NO_LINE_START = frozenset("，。、；：！？）】》」』〕,.;:!?%°'\"”’…—–-)]}．")


class PDFGeneratorTool(Tool):
    """将文本内容保存为 PDF 文件的工具。"""

    name = "pdf_generator"
    description = "将文献综述按中文学术论文体例保存为 PDF，支持中英文混排。"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要保存的文本内容（Markdown 格式）"},
            "output_path": {"type": "string", "description": "PDF 文件保存路径（含文件名，如 output/review.pdf）"},
            "topic": {"type": "string", "description": "研究主题，用于补全文首题名"},
        },
        "required": ["text", "output_path"],
    }

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import fpdf  # noqa: F401
            self._fpdf_available = True
        except ImportError:
            self._fpdf_available = False

    def _markdown_to_pdf(self, text: str, output_path: str, topic: str = "") -> str:
        try:
            import fpdf  # noqa: F401
        except ImportError:
            return "ERROR: fpdf2 未安装，请执行: pip install fpdf2"

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        import logging
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

        markdown = polish_review_markdown(text, topic=topic)
        pdf = _review_pdf_class()()
        pdf.set_auto_page_break(auto=True, margin=_MARGIN_MM)
        pdf.set_margins(left=_MARGIN_MM, top=_MARGIN_MM, right=_MARGIN_MM)
        hei, song = _register_cjk_fonts(pdf)
        times = _register_times(pdf)
        pdf.hei_family = hei
        pdf.song_family = song
        pdf.times_family = times
        pdf.add_page()
        pdf.set_title(_first_heading(markdown) or "文献综述")

        in_refs = False
        for kind, content in _iter_blocks(markdown):
            if kind == "h1":
                _write_title(pdf, hei, content)
                in_refs = False
            elif kind == "h2":
                in_refs = "参考文献" in content
                _write_heading(pdf, hei, content, _H1_PT, before=5, after=2.5)
            elif kind == "h3":
                _write_heading(pdf, hei, content, _H2_PT, before=3.5, after=1.5)
            elif kind == "ref" or (in_refs and kind == "p"):
                _write_flow(pdf, content, _REF_PT, _REF_H_MM, indent=False)
            elif kind == "li":
                _write_flow(pdf, content, _BODY_PT, _BODY_H_MM, indent=False)
            else:
                _write_flow(pdf, content, _BODY_PT, _BODY_H_MM, indent=True)

        pdf.output(output_path)
        return f"PDF 已保存到: {output_path}"

    def run(self, tool_input: dict[str, Any]) -> str:
        text = str(tool_input.get("text", "")).strip()
        output_path = str(tool_input.get("output_path", "")).strip()
        topic = str(tool_input.get("topic", "")).strip()

        if not text:
            return "ERROR: text 参数不能为空"
        if not output_path:
            return "ERROR: output_path 参数不能为空"
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"
        return self._markdown_to_pdf(text, output_path, topic=topic)


def _review_pdf_class():
    from fpdf import FPDF

    class ReviewPDF(FPDF):
        hei_family = "Helvetica"
        song_family = "Helvetica"
        times_family = "Times"

        def footer(self):
            self.set_y(-15)
            try:
                self.set_font(self.song_family, "", _FOOT_PT)
            except Exception:
                self.set_font("Helvetica", "", _FOOT_PT)
            self.cell(0, 8, str(self.page_no()), align="C")

    return ReviewPDF


def _register_cjk_fonts(pdf) -> tuple[str, str]:
    hei = _try_add_font(pdf, "Hei", _HEI_CANDIDATES)
    song = _try_add_font(pdf, "Song", _SONG_CANDIDATES)
    if not hei and not song:
        fallback = _try_add_font(pdf, "CJK", _HEI_CANDIDATES + _SONG_CANDIDATES)
        family = fallback or "Helvetica"
        return family, family
    if not hei:
        hei = song
    if not song:
        song = hei
    return hei, song


def _register_times(pdf) -> str:
    family = _try_add_font(pdf, "TimesNR", _TIMES_CANDIDATES)
    return family or "Times"


def _try_add_font(pdf, family: str, candidates: list[str]) -> str:
    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            pdf.add_font(family, "", path)
            try:
                pdf.add_font(family, "B", path)
            except Exception:
                pass
            return family
        except Exception:
            continue
    return ""


def _first_heading(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    return ""


_LATIN_FIX = str.maketrans({
    "ı": "i",
    "İ": "I",
    "ö": "o",
    "Ö": "O",
    "ü": "u",
    "Ü": "U",
    "ä": "a",
    "Ä": "A",
    "ñ": "n",
    "Ñ": "N",
    "ç": "c",
    "Ç": "C",
    "ş": "s",
    "Ş": "S",
    "ğ": "g",
    "Ğ": "G",
    "\u00a0": " ",
})


def _soften(text: str) -> str:
    """去掉控制符；不在 URL 里插入空格（否则会变成 http: // arxiv. org）。"""
    s = (text or "").replace("\t", " ").replace("\r", " ").replace("\x00", "")
    s = s.replace("\u200b", "").replace("&amp;", "&")
    s = s.translate(_LATIN_FIX)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return repair_broken_identifiers(s.strip())


def _write_title(pdf, font: str, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font(font, "", _TITLE_PT)
    pdf.ln(2)
    _multi(pdf, font, _TITLE_PT, 10, text, align="C")
    pdf.ln(8)


def _write_heading(pdf, font: str, text: str, size: float, before: float, after: float) -> None:
    pdf.ln(before)
    pdf.set_x(pdf.l_margin)
    _multi(pdf, font, size, max(size * 0.6, 7.5), text, align="L")
    pdf.ln(after)


def _write_flow(pdf, text: str, size: float, h: float, indent: bool) -> None:
    """顶格排版，仅首行缩进 2 字；英文 Times New Roman；标点不出现在行首。"""
    payload = _soften(text)
    if not payload:
        return
    song = getattr(pdf, "song_family", "Helvetica")
    times = getattr(pdf, "times_family", "Times")
    tokens = list(_flow_tokens(payload))
    pdf.set_font(song, "", size)
    pdf.set_x(pdf.l_margin + (_INDENT_MM if indent else 0))
    i = 0
    while i < len(tokens):
        kind, token = tokens[i]
        follow: list[tuple[str, str]] = []
        j = i + 1
        while j < len(tokens) and tokens[j][1] in _NO_LINE_START:
            follow.append(tokens[j])
            j += 1
        cluster = [(kind, token), *follow]
        cluster_w = 0.0
        for ck, ct in cluster:
            cluster_w += _token_width(pdf, times if ck == "en" else song, size, ct)
        remaining = pdf.w - pdf.r_margin - pdf.x
        pad = float(getattr(pdf, "c_margin", 1.0) or 1.0) * 2 + 1.5
        if cluster_w + pad > remaining and pdf.x > pdf.l_margin + 0.8:
            pdf.ln(h)
            pdf.set_x(pdf.l_margin)
        _write_token(pdf, times if kind == "en" else song, song, size, h, token)
        for ck, ct in follow:
            _write_punct_stuck(pdf, times if ck == "en" else song, song, size, h, ct)
        i = j
    pdf.ln(h)
    pdf.ln(1.6)


def _token_width(pdf, font: str, size: float, token: str) -> float:
    try:
        pdf.set_font(font, "", size)
        return pdf.get_string_width(token)
    except Exception:
        return 0.0


def _write_punct_stuck(pdf, font: str, fallback: str, size: float, h: float, token: str) -> None:
    """标点跟在前一字后面，即使用 cell 略微超出右边界也不换行。"""
    try:
        pdf.set_font(font, "", size)
    except Exception:
        pdf.set_font(fallback, "", size)
    try:
        w = pdf.get_string_width(token)
        pdf.cell(w, h, token)
    except Exception:
        pdf.write(h, token)


def _write_token(pdf, font: str, fallback: str, size: float, h: float, token: str) -> None:
    try:
        pdf.set_font(font, "", size)
        pdf.write(h, token)
    except Exception:
        pdf.set_font(fallback, "", size)
        pdf.write(h, token)


def _flow_tokens(text: str):
    """中文按字走，英文单词 / [1] / URL 整段走，避免半截换行。"""
    for match in _FLOW_TOKEN.finditer(text or ""):
        token = match.group(1)
        if token.startswith("http://") or token.startswith("https://"):
            parts = token.split("/")
            for i, part in enumerate(parts):
                piece = part if i == 0 else "/" + part
                if piece:
                    yield "en", piece
            continue
        if re.match(r"^\[\d+\]$", token) or re.match(r"^[A-Za-z0-9]", token):
            yield "en", token
        else:
            yield "cjk", token


def _multi(pdf, font: str, size: float, h: float, text: str, align: str) -> None:
    payload = _soften(text)
    if not payload:
        return
    pdf.set_font(font, "", size)
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(0, h, payload, align=align)
        return
    except Exception:
        pass
    try:
        pdf.multi_cell(0, h, payload, align="L")
        return
    except Exception:
        pass
    pdf.set_font("Helvetica", "", max(int(size) - 2, 8))
    ascii_only = payload.encode("ascii", "replace").decode("ascii")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, h, ascii_only[:2000], align="L")


def _iter_blocks(text: str):
    """把 Markdown 收成标题 / 段落 / 参考文献条目（一条文献一行）。"""
    para: list[str] = []
    in_refs = False

    def flush():
        if not para:
            return
        blob = " ".join(para)
        para.clear()
        if in_refs:
            for item in split_reference_entries(blob):
                yield ("ref", item)
        else:
            yield ("p", blob)

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            yield from flush()
            continue
        if line in ("---", "***"):
            yield from flush()
            continue
        if line.startswith("#"):
            yield from flush()
            hashes = len(line) - len(line.lstrip("#"))
            title = clean_inline_markdown(line.lstrip("#"))
            level = 1 if hashes <= 1 else (2 if hashes == 2 else 3)
            if level == 2 and "参考文献" in title:
                in_refs = True
            elif level <= 2:
                in_refs = False
            yield (f"h{level}", title)
            continue
        clean = clean_inline_markdown(line)
        if in_refs:
            if re.match(r"^\[\d+\]", clean):
                yield from flush()
                for item in split_reference_entries(clean):
                    yield ("ref", item)
            else:
                para.append(clean)
            continue
        if re.match(r"^[-*]\s+", clean) or re.match(r"^\d+\.\s+", clean):
            yield from flush()
            yield ("li", clean)
            continue
        para.append(clean)
    yield from flush()
