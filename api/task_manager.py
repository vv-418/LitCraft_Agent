"""
作用：后台任务管理器，负责在单独的线程中运行 Agent 任务，
让 FastAPI 可以立即返回 task_id，前端再通过轮询获取进度和结果。

核心设计：
- 使用 ThreadPoolExecutor 避免阻塞 FastAPI 事件循环
- 任务状态存储在内存字典中（重启后丢失，后续可改为 SQLite 持久化）
- 支持并发运行多个 Agent 任务
"""

import os
import sys
import json
import uuid
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# 确保项目根目录在 sys.path 中，以便导入项目模块
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 初始化日志系统，必须在导入其他项目模块之前执行，
# 否则 scholarly 等第三方库会调用 basicConfig() 创建额外日志文件。
import utils.logger  # noqa: F401

from agent.langgraph_agent import LangGraphAgent
from llm_client import LitCraftAgentsLLM
from tools.registry import ToolRegistry
from tools.search import ArxivSearchTool, SemanticScholarTool, GoogleScholarTool, MultiSourceSearchTool
from tools.paper_downloader import PaperDownloaderTool
from tools.pdf_parser import PDFParserTool
from tools.text_chunker import TextChunkerTool
from tools.vector_store import VectorStoreTool
from tools.advanced_retrieval import AdvancedRetrieval
from tools.pdf_generator import PDFGeneratorTool
from utils.logger import get_logger

logger = get_logger("task_manager")


# 作用：创建大模型客户端、工具注册表，并组装成一个 LangGraphAgent。
# 注意：此函数会在每次新任务时被调用，确保每个任务有独立的工具实例。
def _build_agent() -> LangGraphAgent:
    """创建 LLM、注册工具，并组装成一个 LangGraphAgent。"""
    llm = LitCraftAgentsLLM()

    tools = ToolRegistry()

    # 第0步：多源搜索（arXiv + Semantic Scholar + Google Scholar）
    tools.register(MultiSourceSearchTool(max_results=30, auto_index=True))

    # 第1步：搜索论文（arXiv）
    tools.register(ArxivSearchTool(max_results=10, initial_delay=5.0, max_retries=5))

    # 第1b步：Semantic Scholar（补充）
    tools.register(SemanticScholarTool(max_results=10))

    # 第1c步：Google Scholar（备选）
    tools.register(GoogleScholarTool(max_results=10))

    # 第2步：下载论文（PDF）
    tools.register(PaperDownloaderTool(storage_path="./storage/papers", max_retries=3, timeout=30))

    # 第3步：解析论文（提取文本）
    tools.register(PDFParserTool(storage_path="./storage/papers"))

    # 第4步：分块文本（用于向量化）
    tools.register(TextChunkerTool(chunk_size=512, chunk_overlap=128))

    # 第5步：存储向量（构建向量数据库）
    vector_store = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
    tools.register(vector_store)

    # 第6步：高级检索（HyDE + MQE 混合检索）
    tools.register(AdvancedRetrieval(
        vector_store=vector_store,
        llm=llm,
        num_hypotheses=3,
        num_query_variants=3
    ))

    return LangGraphAgent(llm=llm, tools=tools, max_steps=20)


class TaskManager:
    """后台任务管理器：管理多个 Agent 综述任务的异步执行。"""

    # 作用：初始化管理器，创建空的任务字典和线程池。
    def __init__(self, max_workers: int = 2) -> None:
        """初始化任务管理器。
        
        Args:
            max_workers: 最大并发任务数（默认 2，避免同时消耗过多 API 额度）
        """
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # 作用：启动一个新的综述任务，返回 task_id。
    def start_task(self, topic: str, year_from: str = "", save_pdf: bool = False) -> str:
        """在后台线程中启动一个 Agent 综述任务。
        
        Args:
            topic: 研究主题
            year_from: 年份过滤条件
            save_pdf: 是否自动生成 PDF
        
        Returns:
            新任务的唯一 task_id（UUID 字符串）
        """
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "topic": topic,
                "year_from": year_from,
                "save_pdf": save_pdf,
                "status": "running",
                "final_answer": "",
                "steps": [],
                "pdf_path": None,
                "error": None,
                "created_at": now,
            }

        # 在线程池中异步执行
        logger.info("提交后台任务 | task_id=%s | topic=%.40s | year_from=%s | save_pdf=%s",
                    task_id, topic, year_from, save_pdf)
        self._executor.submit(self._run_task, task_id, topic, year_from, save_pdf)
        return task_id

    # 作用：在后台线程中实际运行 Agent，并更新任务状态。
    def _run_task(self, task_id: str, topic: str, year_from: str, save_pdf: bool) -> None:
        """后台执行 Agent 任务（在 ThreadPoolExecutor 的线程中运行）。
        
        Args:
            task_id: 任务 ID
            topic: 研究主题
            year_from: 年份过滤
            save_pdf: 是否保存 PDF
        """
        try:
            logger.info("开始执行任务 | task_id=%s | topic=%s", task_id, topic)
            agent = _build_agent()
            result = agent.run(topic, year_from=year_from)

            # 构建步骤列表（截断长 observation）
            steps_data = []
            for step in result.steps:
                obs = str(step.observation) if step.observation else None
                if obs and len(obs) > 500:
                    obs = obs[:500] + f"... [已截断，共 {len(obs)} 字符]"
                steps_data.append({
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": obs,
                })

            # 如果 save_pdf=True，自动生成 PDF
            pdf_path = None
            if save_pdf and result.final_answer:
                output_dir = "./output"
                os.makedirs(output_dir, exist_ok=True)
                safe_topic = topic.strip().replace(" ", "_")[:30]
                output_filename = f"review_{task_id[:8]}_{safe_topic}.pdf"
                output_path = os.path.join(output_dir, output_filename)

                pdf_gen = PDFGeneratorTool()
                pdf_result = pdf_gen.run({
                    "text": result.final_answer,
                    "output_path": output_path,
                })
                if not pdf_result.startswith("ERROR"):
                    pdf_path = output_path

            with self._lock:
                self._tasks[task_id].update({
                    "status": "done",
                    "final_answer": result.final_answer,
                    "steps": steps_data,
                    "pdf_path": pdf_path,
                })

            logger.info("任务完成 | task_id=%s | 步骤数=%d | 答案长度=%d | pdf=%s",
                        task_id, len(result.steps),
                        len(result.final_answer) if result.final_answer else 0,
                        pdf_path or "无")

        except Exception as exc:
            import traceback as tb_module
            tb_str = "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
            logger.error("任务执行失败 | task_id=%s | error=%s",
                         task_id, str(exc), exc_info=True)
            with self._lock:
                self._tasks[task_id].update({
                    "status": "error",
                    "error": str(exc),
                    "traceback": tb_str,
                })

    # 作用：查询某个任务的状态（running / done / error）。
    def get_status(self, task_id: str) -> str | None:
        """获取任务当前状态。
        
        Args:
            task_id: 任务 ID
        
        Returns:
            "running" / "done" / "error" 之一；如果任务不存在返回 None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return task["status"] if task else None

    # 作用：获取某个任务的执行结果（包含 final_answer 和 steps）。
    def get_result(self, task_id: str) -> dict[str, Any] | None:
        """获取任务的完整结果。
        
        Args:
            task_id: 任务 ID
        
        Returns:
            任务数据字典，如果任务不存在返回 None
        """
        with self._lock:
            return self._tasks.get(task_id)

    # 作用：获取某个任务的执行步骤列表（供前端轮询更新展示）。
    def get_steps(self, task_id: str) -> list[dict[str, Any]]:
        """获取任务的执行步骤（可用于前端实时展示进度）。
        
        Args:
            task_id: 任务 ID
        
        Returns:
            步骤列表（可能部分完成），任务不存在时返回空列表
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return task["steps"] if task else []

    # 作用：获取历史任务列表（按创建时间倒序排列）。
    def list_tasks(self) -> list[dict[str, Any]]:
        """获取所有历史任务的摘要列表。
        
        Returns:
            按创建时间倒序排列的任务摘要列表
        """
        with self._lock:
            tasks = [
                {
                    "task_id": t["task_id"],
                    "topic": t["topic"],
                    "status": t["status"],
                    "created_at": t["created_at"],
                }
                for t in self._tasks.values()
            ]
        return sorted(tasks, key=lambda x: x["created_at"], reverse=True)
