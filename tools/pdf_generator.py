"""PDF 生成工具：将文献综述文本保存为 PDF 文件，支持中文/英文混排。"""

import os
from typing import Any

from tools.base import Tool


class PDFGeneratorTool(Tool):
    """将文本内容保存为 PDF 文件的工具。"""

    name = "pdf_generator"
    description = "将文献综述文本保存为 PDF 文件，支持中英文混排。"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要保存的文本内容（Markdown 格式）"},
            "output_path": {"type": "string", "description": "PDF 文件保存路径（含文件名，如 output/review.pdf）"},
        },
        "required": ["text", "output_path"],
    }

    def __init__(self):
        """初始化 PDF 生成器。"""
        self._check_dependencies()

    def _check_dependencies(self):
        """检查 fpdf2 是否安装。"""
        try:
            import fpdf
            self._fpdf_available = True
        except ImportError:
            self._fpdf_available = False

    def _markdown_to_pdf(self, text: str, output_path: str) -> str:
        """将 Markdown 文本转换为 PDF（去掉 Markdown 标记，保留纯文本结构）。"""
        try:
            from fpdf import FPDF
        except ImportError:
            return "ERROR: fpdf2 未安装，请执行: pip install fpdf2"

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_margins(left=15, top=15, right=15)
        pdf.add_page()

        # 尝试使用支持中文的字体
        font_loaded = False

        # 候选字体路径（按优先级）
        font_candidates = [
            # Windows 简体中文字体
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyh.ttf",        # 微软雅黑（单个 ttf）
            # Windows 繁体中文字体
            "C:/Windows/Fonts/msjh.ttc",        # 微软正黑体
            "C:/Windows/Fonts/mingliu.ttc",     # 细明体
            # 系统 fallback
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",  # Linux 文泉驿
            "/System/Library/Fonts/PingFang.ttc",              # macOS 苹方
        ]

        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    pdf.add_font("CustomFont", "", font_path, uni=True)
                    pdf.add_font("CustomFont", "B", font_path, uni=True)
                    font_loaded = True
                    break
                except Exception:
                    continue

        if not font_loaded:
            # 如果找不到中文字体，尝试注册系统上的任何 ttf 字体
            # 遍历 Windows 字体目录
            win_fonts_dir = "C:/Windows/Fonts"
            if os.path.exists(win_fonts_dir):
                for root, dirs, files in os.walk(win_fonts_dir):
                    for f in files:
                        if f.endswith(".ttf") or f.endswith(".ttc"):
                            font_path = os.path.join(root, f)
                            try:
                                pdf.add_font("CustomFont", "", font_path, uni=True)
                                pdf.add_font("CustomFont", "B", font_path, uni=True)
                                font_loaded = True
                                break
                            except Exception:
                                continue
                    if font_loaded:
                        break

        if not font_loaded:
            # 终极 fallback：用内置字体（不支持中文，但至少能出英文 PDF）
            font_family = "Courier"
        else:
            font_family = "CustomFont"

        def _safe_cell(pdf_obj, font, style, size, text, h):
            """带异常兜底和 x 位置重置的 multi_cell。"""
            pdf_obj.set_x(15)
            pdf_obj.set_font(font, style, size)
            try:
                pdf_obj.multi_cell(0, h, text)
            except Exception:
                cleaned = text.replace("\t", " ").replace("\r", " ").replace("\x00", "")
                try:
                    pdf_obj.set_font(font, style, max(size - 1, 6))
                    pdf_obj.set_x(15)
                    pdf_obj.multi_cell(0, h, cleaned)
                except Exception:
                    pdf_obj.set_font("Courier", "", 8)
                    pdf_obj.set_x(15)
                    pdf_obj.multi_cell(0, h, cleaned[:1000])

        # 清理 markdown 标记，保留纯文本结构
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()

            # 跳过空行，但保留段落间距
            if not stripped:
                pdf.ln(4)
                continue

            # 判断是否是标题
            is_heading = False
            if stripped.startswith("## "):
                heading_text = stripped[3:].strip()
                is_heading = True
                _safe_cell(pdf, font_family, "B", 14, heading_text, 8)
                pdf.ln(2)
            elif stripped.startswith("# "):
                heading_text = stripped[2:].strip()
                is_heading = True
                _safe_cell(pdf, font_family, "B", 16, heading_text, 10)
                pdf.ln(3)
            elif stripped.startswith("### "):
                heading_text = stripped[4:].strip()
                is_heading = True
                _safe_cell(pdf, font_family, "B", 12, heading_text, 7)
                pdf.ln(1)

            if is_heading:
                continue

            # 普通段落：去掉粗体/斜体标记，保留文字
            clean = stripped.replace("**", "").replace("*", "").replace("`", "")
            # 去掉列表标记
            if clean.startswith("- ") or clean.startswith("* "):
                clean = "  " + clean
            # 数字列表
            import re
            clean = re.sub(r'^\d+\.\s+', '   ', clean)

            # 引用标记
            if clean.startswith("> "):
                clean = "  " + clean[2:]

            # 水平线
            if clean == "---" or clean == "***":
                pdf.ln(2)
                continue

            _safe_cell(pdf, font_family, "", 10, clean, 5)

        pdf.output(output_path)
        return f"PDF 已保存到: {output_path}"

    def run(self, tool_input: dict[str, Any]) -> str:
        """执行 PDF 生成。"""
        text = str(tool_input.get("text", "")).strip()
        output_path = str(tool_input.get("output_path", "")).strip()

        if not text:
            return "ERROR: text 参数不能为空"
        if not output_path:
            return "ERROR: output_path 参数不能为空"

        # 如果路径没有 .pdf 后缀，自动添加
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"

        return self._markdown_to_pdf(text, output_path)
