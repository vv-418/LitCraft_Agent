# 作用：实现文本分块工具，将长文本分成固定大小的块，支持重叠。
import json
from typing import Any, List, Dict
import re

from tools.base import Tool


class TextChunk(dict):
    """单个文本块。"""
    def __init__(self, content: str, chunk_id: int, source: str = "", metadata: Dict[str, Any] = None):
        self["content"] = content
        self["chunk_id"] = chunk_id
        self["source"] = source
        self["metadata"] = metadata or {}


class TextChunkerTool(Tool):
    """将长文本分成可管理的块，支持重叠。"""

    name = "text_chunker"
    description = "Split long texts into chunks with optional overlap for better context preservation."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text content to chunk"},
            "chunk_size": {"type": "integer", "description": "Size of each chunk in characters (default: 512)"},
            "chunk_overlap": {"type": "integer", "description": "Overlap between chunks in characters (default: 128)"},
            "source": {"type": "string", "description": "Source identifier (e.g., filename, paper title)"},
            "metadata": {"type": "object", "description": "Additional metadata to attach to chunks"},
        },
        "required": ["text"],
    }

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """初始化 TextChunkerTool。
        
        Args:
            chunk_size: 每个块的大小（字符数）
            chunk_overlap: 块之间的重叠（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本。
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 简单的句子分割（支持中英文）
        # 匹配句号、问号、感叹号、换行等
        sentences = re.split(r'(?<=[。！？\n])\s*(?=[^\s])|(?<=[.!?])\s+(?=[A-Z])', text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks(self, text: str, sentences: List[str]) -> List[str]:
        """创建带重叠的文本块。
        
        Args:
            text: 原始文本
            sentences: 句子列表
            
        Returns:
            文本块列表
        """
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # 如果添加这个句子会超过块大小，开始新块
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # 计算重叠部分
                overlap_text = current_chunk
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap:]
                
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """简单的字符级分块（当句子分割失败时）。
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def run(self, tool_input: dict[str, Any]) -> str:
        """将文本分块。
        
        Args:
            tool_input: 包含文本和分块参数
            
        Returns:
            JSON 字符串，包含分块结果
        """
        text = str(tool_input.get("text", "")).strip()
        chunk_size = int(tool_input.get("chunk_size", self.chunk_size))
        chunk_overlap = int(tool_input.get("chunk_overlap", self.chunk_overlap))
        source = str(tool_input.get("source", "")).strip()
        metadata = tool_input.get("metadata", {})
        
        if not text:
            return json.dumps({"error": "Text cannot be empty"}, ensure_ascii=False)
        
        if chunk_size <= 0:
            return json.dumps({"error": "chunk_size must be positive"}, ensure_ascii=False)
        
        if chunk_overlap >= chunk_size:
            return json.dumps(
                {"error": "chunk_overlap must be smaller than chunk_size"},
                ensure_ascii=False
            )
        
        try:
            print("[CHUNK] 开始分块文本...")
            print(f"   原始长度: {len(text):,} 字符")
            print(f"   块大小: {chunk_size}, 重叠: {chunk_overlap}")
            
            # 尝试按句子分割
            sentences = self._split_by_sentences(text)
            
            if len(sentences) > 1:
                # 按句子分割
                chunks = self._create_chunks(text, sentences)
            else:
                # 如果只有一个句子或分割失败，使用字符级分块
                chunks = self._simple_chunk(text)
            
            # 过滤太短的块
            chunks = [c for c in chunks if len(c) > 10]
            
            # 创建块对象
            chunk_objects = []
            for idx, chunk_text in enumerate(chunks):
                chunk_obj = {
                    "chunk_id": idx,
                    "content": chunk_text,
                    "length": len(chunk_text),
                    "source": source,
                    "metadata": metadata
                }
                chunk_objects.append(chunk_obj)
            
            result = {
                "status": "success",
                "total_chunks": len(chunks),
                "original_length": len(text),
                "total_chunk_length": sum(c["length"] for c in chunk_objects),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "source": source,
                "chunks": chunk_objects
            }
            
            print(f"[OK] 分块成功")
            print(f"   生成块数: {len(chunks)}")
            print(f"   平均块大小: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] 分块失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "error": error_msg,
                },
                ensure_ascii=False
            )
