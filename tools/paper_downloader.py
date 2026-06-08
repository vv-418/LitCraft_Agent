# 作用：实现从 arXiv 下载论文 PDF 的工具，带重试机制和错误处理。
import os
import json
import time
from typing import Any
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools.base import Tool


class PaperDownloaderTool(Tool):
    """从 arXiv 下载论文 PDF 的工具。"""

    name = "paper_downloader"
    description = "Download PDF papers from arXiv URLs and save them locally."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "PDF URL or arXiv URL"},
            "filename": {"type": "string", "description": "Local filename to save as (optional)"},
        },
        "required": ["url"],
    }

    def __init__(self, storage_path: str = "./storage/papers", max_retries: int = 3, timeout: int = 30):
        """初始化 PaperDownloaderTool。
        
        Args:
            storage_path: 论文存储目录
            max_retries: 最大重试次数
            timeout: 下载超时时间（秒）
        """
        self.storage_path = Path(storage_path)
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 创建存储目录（如果不存在）
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _create_session(self) -> requests.Session:
        """创建带有重试策略的 requests 会话。"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置用户代理
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        return session

    def _get_filename_from_url(self, url: str) -> str:
        """从 URL 提取文件名。"""
        # 处理 arXiv URL 格式: https://arxiv.org/pdf/2606.06491v1.pdf
        # 或 https://arxiv.org/abs/2606.06491v1
        if "/pdf/" in url:
            filename = url.split("/pdf/")[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
        elif "/abs/" in url:
            arxiv_id = url.split("/abs/")[-1]
            filename = f"{arxiv_id}.pdf"
        else:
            # 作为备选方案，使用最后一段
            filename = url.split("/")[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
        
        return filename

    def _normalize_url(self, url: str) -> str:
        """规范化 URL，将 arXiv abstract URL 转换为 PDF URL。"""
        url = url.strip()
        
        # 如果是 abs URL，转换为 pdf URL
        if "/abs/" in url:
            arxiv_id = url.split("/abs/")[-1]
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # 如果已经是 PDF URL，直接返回
        if url.endswith(".pdf"):
            return url
        
        # 如果是不完整的 PDF URL，补全
        if "/pdf/" in url:
            return url if url.endswith(".pdf") else url + ".pdf"
        
        return url

    def run(self, tool_input: dict[str, Any]) -> str:
        """下载 PDF 文件。
        
        Args:
            tool_input: 包含 'url' 和可选的 'filename' 字段
            
        Returns:
            JSON 字符串，包含下载结果
        """
        url = str(tool_input.get("url", "")).strip()
        custom_filename = str(tool_input.get("filename", "")).strip()
        
        if not url:
            return json.dumps({"error": "URL cannot be empty"}, ensure_ascii=False)

        # 规范化 URL
        pdf_url = self._normalize_url(url)
        
        # 确定文件名
        if custom_filename:
            filename = custom_filename if custom_filename.endswith(".pdf") else custom_filename + ".pdf"
        else:
            filename = self._get_filename_from_url(pdf_url)
        
        # 构建完整文件路径
        file_path = self.storage_path / filename
        
        # 检查文件是否已存在
        if file_path.exists():
            file_size = file_path.stat().st_size
            return json.dumps(
                {
                    "status": "already_exists",
                    "filename": filename,
                    "path": str(file_path),
                    "size_bytes": file_size,
                    "message": f"File already exists: {filename} ({file_size} bytes)"
                },
                ensure_ascii=False
            )

        # 下载 PDF
        try:
            print(f"[DOWNLOAD] 开始下载: {filename}...")
            print(f"   URL: {pdf_url}")
            
            session = self._create_session()
            
            with session.get(pdf_url, timeout=self.timeout, stream=True) as response:
                response.raise_for_status()
                
                # 获取文件大小
                total_size = int(response.headers.get("content-length", 0))
                
                # 下载文件
                downloaded_size = 0
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"   进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end="\r")
                
                print(f"\n[OK] 下载成功: {filename} ({downloaded_size} bytes)")
                
                return json.dumps(
                    {
                        "status": "success",
                        "filename": filename,
                        "path": str(file_path),
                        "size_bytes": downloaded_size,
                        "url": pdf_url,
                        "message": f"Successfully downloaded: {filename}"
                    },
                    ensure_ascii=False
                )

        except requests.exceptions.Timeout:
            error_msg = f"Download timeout (>{self.timeout}s)"
            print(f"[ERROR] 下载超时: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "url": pdf_url,
                    "filename": filename
                },
                ensure_ascii=False
            )

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {e.response.status_code}: {e.response.reason}"
            print(f"[ERROR] HTTP 错误: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "url": pdf_url,
                    "filename": filename,
                    "http_status": e.response.status_code
                },
                ensure_ascii=False
            )

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            print(f"[ERROR] 请求失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "url": pdf_url,
                    "filename": filename
                },
                ensure_ascii=False
            )

        except IOError as e:
            error_msg = f"File I/O error: {str(e)}"
            print(f"[ERROR] 文件保存错误: {error_msg}")
            # 清理不完整的文件
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "filename": filename
                },
                ensure_ascii=False
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[ERROR] 未知错误: {error_msg}")
            # 清理不完整的文件
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                    "filename": filename
                },
                ensure_ascii=False
            )
