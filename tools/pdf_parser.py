# 作用：实现从 PDF 论文文件中提取文本、元数据和结构化内容的工具。
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdfplumber
from pydantic import BaseModel

from tools.base import Tool


class PageContent(BaseModel):
    """单个页面的内容。"""
    page_num: int
    text: str
    tables: List[Dict[str, Any]] = []


class PaperMetadata(BaseModel):
    """论文的元数据。"""
    filename: str
    total_pages: int
    title: Optional[str] = None
    authors: List[str] = []
    creation_date: Optional[str] = None
    producer: Optional[str] = None


class PDFParserTool(Tool):
    """从 PDF 中提取文本内容和元数据的工具。"""

    name = "pdf_parser"
    description = "Extract text content, metadata, and tables from PDF papers."
    input_schema = {
        "type": "object",
        "properties": {
            "pdf_path": {"type": "string", "description": "Path to PDF file"},
            "extract_tables": {"type": "boolean", "description": "Extract tables from PDF (optional, default: true)"},
            "pages": {"type": "string", "description": "Specific pages to extract (e.g., '1-5', '1,3,5', optional)"},
        },
        "required": ["pdf_path"],
    }

    def __init__(self, storage_path: str = "./storage/papers"):
        """初始化 PDFParserTool。
        
        Args:
            storage_path: 论文存储目录
        """
        self.storage_path = Path(storage_path)

    def _parse_page_range(self, page_spec: str, total_pages: int) -> List[int]:
        """解析页面范围说明。
        
        Args:
            page_spec: 页面说明，例如 '1-5', '1,3,5'
            total_pages: 总页数
            
        Returns:
            页面号列表（0-indexed）
        """
        try:
            pages = []
            
            # 处理逗号分隔的单个页面
            if "," in page_spec:
                for part in page_spec.split(","):
                    page_num = int(part.strip()) - 1  # 转换为0-indexed
                    if 0 <= page_num < total_pages:
                        pages.append(page_num)
            # 处理范围
            elif "-" in page_spec:
                start, end = page_spec.split("-")
                start_page = int(start.strip()) - 1  # 转换为0-indexed
                end_page = int(end.strip()) - 1  # 转换为0-indexed
                pages = list(range(start_page, min(end_page + 1, total_pages)))
            # 单个页面
            else:
                page_num = int(page_spec.strip()) - 1  # 转换为0-indexed
                if 0 <= page_num < total_pages:
                    pages.append(page_num)
            
            return sorted(list(set(pages)))  # 去重并排序
        except Exception as e:
            print(f"[WARN] 页面范围解析失败: {e}，将解析全部页面")
            return list(range(total_pages))

    def _extract_metadata(self, pdf) -> Dict[str, Any]:
        """从 PDF 中提取元数据。
        
        Args:
            pdf: pdfplumber PDF 对象
            
        Returns:
            元数据字典
        """
        metadata = {}
        
        # 获取 PDF 的基本元数据
        if pdf.metadata:
            pdf_meta = pdf.metadata
            metadata["title"] = pdf_meta.get("Title", "")
            metadata["author"] = pdf_meta.get("Author", "")
            metadata["subject"] = pdf_meta.get("Subject", "")
            metadata["creator"] = pdf_meta.get("Creator", "")
            metadata["producer"] = pdf_meta.get("Producer", "")
            metadata["creation_date"] = str(pdf_meta.get("CreationDate", ""))
            metadata["modification_date"] = str(pdf_meta.get("ModDate", ""))
        
        return metadata

    def _clean_text(self, text: str) -> str:
        """清理提取的文本。
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空格和换行
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # 移除空行
        return "\n".join(lines)

    def _extract_first_page_info(self, pdf) -> Dict[str, Any]:
        """从首页提取关键信息（标题、作者等）。
        
        Args:
            pdf: pdfplumber PDF 对象
            
        Returns:
            包含标题、作者等信息的字典
        """
        info = {
            "title": "",
            "authors": [],
            "abstract": ""
        }
        
        try:
            if len(pdf.pages) == 0:
                return info
            
            # 获取首页文本
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            if not text:
                return info
            
            lines = text.split("\n")
            
            # 简单启发式：前几行可能包含标题
            # 标题通常是全大写或较长的文本
            for i, line in enumerate(lines[:20]):
                line = line.strip()
                if not line:
                    continue
                
                # 标题通常较长
                if len(line) > 20 and len(line) < 200:
                    if not info["title"]:
                        info["title"] = line
                        break
            
            # 查找 "ABSTRACT" 关键词并提取摘要
            abstract_start = None
            for i, line in enumerate(lines):
                if "ABSTRACT" in line.upper() or "SUMMARY" in line.upper():
                    abstract_start = i
                    break
            
            if abstract_start is not None:
                abstract_lines = []
                for i in range(abstract_start + 1, min(abstract_start + 15, len(lines))):
                    line = lines[i].strip()
                    if line and not any(keyword in line.upper() for keyword in ["INTRODUCTION", "1.", "KEYWORDS"]):
                        abstract_lines.append(line)
                    elif any(keyword in line.upper() for keyword in ["INTRODUCTION", "1."]):
                        break
                info["abstract"] = " ".join(abstract_lines)[:500]  # 限制摘要长度
            
            return info
        except Exception as e:
            print(f"[WARN] 首页信息提取失败: {e}")
            return info

    def run(self, tool_input: dict[str, Any]) -> str:
        """解析 PDF 文件并提取内容。
        
        Args:
            tool_input: 包含 'pdf_path' 和可选参数的字典
            
        Returns:
            JSON 字符串，包含提取的内容
        """
        pdf_path_str = str(tool_input.get("pdf_path", "")).strip()
        extract_tables = tool_input.get("extract_tables", True)
        pages_spec = str(tool_input.get("pages", "")).strip()
        
        if not pdf_path_str:
            return json.dumps({"error": "PDF path cannot be empty"}, ensure_ascii=False)
        
        # 处理相对路径：多策略尝试
        pdf_path = Path(pdf_path_str)
        if not pdf_path.is_absolute():
            found = None
            
            # 策略1: storage_path / pdf_path_str（处理纯文件名的情况）
            candidate = self.storage_path / pdf_path_str
            if candidate.exists():
                found = candidate
            
            # 策略2: storage_path / just_basename（处理 agent 传入了路径前缀的情况）
            if found is None:
                basename = pdf_path.name
                candidate2 = self.storage_path / basename
                if candidate2.exists():
                    found = candidate2
            
            # 策略3: 使用 CWD 相对路径直接尝试
            if found is None and pdf_path.exists():
                found = pdf_path
            
            if found is None:
                return json.dumps(
                    {"error": f"PDF file not found: {pdf_path_str}"},
                    ensure_ascii=False
                )
            pdf_path = found
        elif not pdf_path.exists():
            return json.dumps(
                {"error": f"PDF file not found: {pdf_path_str}"},
                ensure_ascii=False
            )
        
        try:
            # ── 支持 .txt 文件（HTML 提取的论文正文） ──────────
            if pdf_path.suffix.lower() in (".txt", ".html"):
                print(f"[TXT] 读取文本文件: {pdf_path.name}...")
                with open(pdf_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                return json.dumps({
                    "filename": pdf_path.name,
                    "format": "text",
                    "pages_count": 1,
                    "total_chars": len(text),
                    "metadata": {},
                    "first_page_info": {"title": pdf_path.stem, "authors": [], "abstract": ""},
                    "content": text,
                }, ensure_ascii=False)

            print(f"[PDF] 开始解析 PDF: {pdf_path.name}...")
            
            with pdfplumber.open(str(pdf_path)) as pdf:
                # 提取元数据
                print(f"   [META] 提取元数据...")
                metadata = self._extract_metadata(pdf)
                first_page_info = self._extract_first_page_info(pdf)
                
                # 确定要处理的页面
                total_pages = len(pdf.pages)
                if pages_spec:
                    page_indices = self._parse_page_range(pages_spec, total_pages)
                else:
                    page_indices = list(range(total_pages))
                
                print(f"   [PAGE] 解析 {len(page_indices)} 页...")
                
                # 提取页面内容
                pages_content = []
                full_text = []
                
                for idx, page_idx in enumerate(page_indices):
                    page = pdf.pages[page_idx]
                    
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        text = self._clean_text(text)
                        full_text.append(text)
                    
                    # 提取表格
                    tables = []
                    if extract_tables:
                        try:
                            page_tables = page.extract_tables()
                            if page_tables:
                                tables = [
                                    {
                                        "rows": table,
                                        "row_count": len(table),
                                        "col_count": len(table[0]) if table else 0
                                    }
                                    for table in page_tables
                                ]
                        except Exception as e:
                            pass  # 表格提取失败，忽略
                    
                    pages_content.append({
                        "page_num": page_idx + 1,
                        "text": text or "",
                        "table_count": len(tables),
                        "tables": tables if extract_tables else []
                    })
                    
                    # 显示进度
                    progress = (idx + 1) / len(page_indices) * 100
                    print(f"   进度: {progress:.1f}% ({idx + 1}/{len(page_indices)} 页)", end="\r")
                
                # 汇总结果
                result = {
                    "status": "success",
                    "filename": pdf_path.name,
                    "total_pages": total_pages,
                    "extracted_pages": len(page_indices),
                    "metadata": {
                        "title": first_page_info.get("title") or metadata.get("title", ""),
                        "authors": first_page_info.get("authors") or metadata.get("author", "").split(";") if metadata.get("author") else [],
                        "abstract": first_page_info.get("abstract", ""),
                        "creation_date": metadata.get("creation_date", ""),
                        "producer": metadata.get("producer", ""),
                    },
                    "content": {
                        "pages": pages_content,
                        "full_text": "\n\n".join(full_text),
                        "character_count": sum(len(p.get("text", "")) for p in pages_content),
                        "table_count": sum(p.get("table_count", 0) for p in pages_content),
                    }
                }
                
                print(f"\n[OK] PDF 解析成功: {pdf_path.name}")
                print(f"   [PAGE] 总页数: {total_pages}")
                print(f"   [CHARS] 字符数: {result['content']['character_count']}")
                print(f"   [TABLES] 表格数: {result['content']['table_count']}")
                
                return json.dumps(result, ensure_ascii=False, indent=2)
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] PDF 解析失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "filename": pdf_path.name
                },
                ensure_ascii=False
            )
