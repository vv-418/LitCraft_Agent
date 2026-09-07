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
from tools.search import (
    ArxivSearchTool,
    SemanticScholarTool,
    GoogleScholarTool,
    OpenAlexTool,
    CrossrefTool,
    EuropePMCTool,
    MultiSourceSearchTool,
)
from tools.paper_downloader import PaperDownloaderTool
from tools.pdf_parser import PDFParserTool
from tools.text_chunker import TextChunkerTool
from tools.vector_store import VectorStoreTool
from tools.advanced_retrieval import AdvancedRetrieval
from tools.pdf_generator import PDFGeneratorTool
from utils.logger import get_logger
from utils.step_view import serialize_step

logger = get_logger("task_manager")


# 作用：创建大模型客户端、工具注册表，并组装成一个 LangGraphAgent。
# 注意：此函数会在每次新任务时被调用，确保每个任务有独立的工具实例。
def _build_agent(
    per_source_limit: int = 10,
    final_limit: int = 10,
    on_step=None,
    on_progress=None,
    papers_dir: str = "",
    figures_dir: str = "",
) -> LangGraphAgent:
    """创建 LLM、注册工具，并组装成一个 LangGraphAgent。"""
    llm = LitCraftAgentsLLM()

    tools = ToolRegistry()
    per_source = max(1, min(int(per_source_limit or 10), 50))
    final_n = max(1, min(int(final_limit or 10), 50))

    # 第0步：多源搜索（OpenAlex / Crossref / Europe PMC / arXiv / Semantic Scholar）
    tools.register(MultiSourceSearchTool(
        max_results=final_n,
        per_source_limit=per_source,
        auto_index=True,
        papers_dir=papers_dir,
        figures_dir=figures_dir,
    ))

    tools.register(ArxivSearchTool(max_results=per_source, initial_delay=5.0, max_retries=5))
    tools.register(SemanticScholarTool(max_results=per_source))
    tools.register(GoogleScholarTool(max_results=per_source))
    tools.register(OpenAlexTool(max_results=per_source))
    tools.register(CrossrefTool(max_results=per_source))
    tools.register(EuropePMCTool(max_results=per_source))

    # 第2步：下载论文（PDF）——直接写入任务 lit_source
    tools.register(PaperDownloaderTool(
        storage_path=papers_dir,
        max_retries=3,
        timeout=30,
    ))

    # 第3步：解析论文（提取文本）
    tools.register(PDFParserTool(storage_path=papers_dir, figures_path=figures_dir))

    # 第4步：分块文本（用于向量化）
    tools.register(TextChunkerTool(chunk_size=512, chunk_overlap=128))

    # 第5步：存储向量（构建向量数据库）
    vector_store = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
    tools.register(vector_store)

    # 第6步：证据检索（默认 Dense；RETRIEVAL_MODE=hybrid 开 RRF+重排）
    tools.register(AdvancedRetrieval(
        vector_store=vector_store,
        llm=llm,
        num_hypotheses=3,
        num_query_variants=3
    ))

    return LangGraphAgent(
        llm=llm,
        tools=tools,
        max_steps=20,
        on_step=on_step,
        on_progress=on_progress,
    )


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
    def start_task(
        self,
        topic: str,
        year_from: str = "",
        save_pdf: bool = False,
        per_source_limit: int = 10,
        final_limit: int = 10,
        output_dir: str = "",
        papers_output_dir: str = "",
        papers_filename: str = "",
        review_output_dir: str = "",
        review_filename: str = "",
    ) -> str:
        """在后台线程中启动一个 Agent 综述任务。
        
        Args:
            topic: 研究主题
            year_from: 年份过滤条件
            save_pdf: 是否自动生成 PDF
            per_source_limit: 每个学术源单次最多保留篇数
            final_limit: 多源合并后最终对外保留篇数
            output_dir: 兼容旧参数，同时作为论文/综述自定义根目录
            papers_output_dir: 下载论文保存目录
            papers_filename: 下载论文文件名前缀
            review_output_dir: 综述 PDF 保存目录
            review_filename: 综述 PDF 自定义文件名
        
        Returns:
            新任务的唯一 task_id（UUID 字符串）
        """
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        per_source = max(1, min(int(per_source_limit or 10), 50))
        final_n = max(1, min(int(final_limit or 10), 50))
        papers_dir = (papers_output_dir or "").strip()
        review_dir = (review_output_dir or "").strip()
        out_dir = (output_dir or "").strip()

        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "topic": topic,
                "year_from": year_from,
                "save_pdf": save_pdf,
                "output_dir": out_dir,
                "papers_output_dir": papers_dir,
                "papers_filename": (papers_filename or "").strip(),
                "review_output_dir": review_dir,
                "review_filename": (review_filename or "").strip(),
                "per_source_limit": per_source,
                "final_limit": final_n,
                "status": "running",
                "final_answer": "",
                "steps": [],
                "step_count": 0,
                "pdf_path": None,
                "output_folder": None,
                "papers_folder": None,
                "review_folder": None,
                "error": None,
                "created_at": now,
            }

        # 在线程池中异步执行
        logger.info(
            "提交后台任务 | task_id=%s | topic=%.40s | year_from=%s | save_pdf=%s | per_source=%s | final=%s | papers_dir=%s | review_dir=%s",
            task_id, topic, year_from, save_pdf, per_source, final_n,
            papers_dir or out_dir or "default",
            review_dir or out_dir or "default",
        )
        self._executor.submit(
            self._run_task,
            task_id,
            topic,
            year_from,
            save_pdf,
            per_source,
            final_n,
            out_dir,
            papers_dir,
            (papers_filename or "").strip(),
            review_dir,
            (review_filename or "").strip(),
        )
        return task_id

    # 作用：在后台线程中实际运行 Agent，并更新任务状态。
    def _run_task(
        self,
        task_id: str,
        topic: str,
        year_from: str,
        save_pdf: bool,
        per_source_limit: int = 10,
        final_limit: int = 10,
        output_dir: str = "",
        papers_output_dir: str = "",
        papers_filename: str = "",
        review_output_dir: str = "",
        review_filename: str = "",
    ) -> None:
        """后台执行 Agent 任务（在 ThreadPoolExecutor 的线程中运行）。
        
        Args:
            task_id: 任务 ID
            topic: 研究主题
            year_from: 年份过滤
            save_pdf: 是否保存 PDF
            per_source_limit: 单源篇数
            final_limit: 最终保留篇数
            output_dir: 兼容旧的单一保存根目录
            papers_output_dir: 下载论文保存目录
            papers_filename: 下载论文文件名前缀
            review_output_dir: 综述保存目录
            review_filename: 综述自定义文件名
        """
        try:
            logger.info("开始执行任务 | task_id=%s | topic=%s", task_id, topic)

            def _on_step(step) -> None:
                """每完成一步就写入任务字典，供前端轮询。"""
                payload = serialize_step(step)
                with self._lock:
                    task = self._tasks.get(task_id)
                    if not task or task.get("status") != "running":
                        return
                    # 去掉「执行中」占位，换上真实结果
                    steps = [s for s in task.get("steps", []) if not s.get("pending")]
                    steps.append(payload)
                    task["steps"] = steps
                    task["step_count"] = len(steps)

            def _on_progress(info: dict) -> None:
                """工具开始执行时写入占位步骤，避免长搜索时前端一直 0 步。"""
                from utils.step_view import build_step_summary
                from agent.models import AgentStep

                action = str(info.get("action") or "")
                pending_step = AgentStep(
                    thought=str(info.get("thought") or ""),
                    action=action or None,
                    action_input=info.get("action_input") or {},
                    observation="（执行中，请稍候…）",
                )
                summary = build_step_summary(pending_step)
                summary["kind"] = "pending"
                summary["headline"] = f"正在执行「{summary.get('title') or action}」…"
                payload = {
                    "thought": pending_step.thought,
                    "action": pending_step.action,
                    "action_input": pending_step.action_input,
                    "observation": pending_step.observation,
                    "summary": summary,
                    "pending": True,
                }
                with self._lock:
                    task = self._tasks.get(task_id)
                    if not task or task.get("status") != "running":
                        return
                    steps = [s for s in task.get("steps", []) if not s.get("pending")]
                    steps.append(payload)
                    task["steps"] = steps
                    task["step_count"] = len(steps)

            from utils.review_output import resolve_papers_layout, save_run_outputs

            started = datetime.now()
            query_folder, lit_source, figures = resolve_papers_layout(
                topic,
                papers_output_dir=papers_output_dir,
                output_dir=output_dir,
                when=started,
            )
            logger.info(
                "论文直接保存到 | root=%s | lit_source=%s",
                query_folder, lit_source,
            )

            agent = _build_agent(
                per_source_limit=per_source_limit,
                final_limit=final_limit,
                on_step=_on_step,
                on_progress=_on_progress,
                papers_dir=str(lit_source),
                figures_dir=str(figures),
            )
            result = agent.run(
                topic,
                year_from=year_from,
                per_source_limit=per_source_limit,
                final_limit=final_limit,
            )

            # 完成后用完整序列化结果覆盖（保证 summary / 截断一致）
            steps_data = [serialize_step(step) for step in result.steps]

            pdf_gen = PDFGeneratorTool()

            def _write_review(output_path: str) -> str:
                return pdf_gen.run({
                    "text": result.final_answer,
                    "output_path": output_path,
                })

            saved = save_run_outputs(
                topic=topic,
                steps=result.steps,
                final_answer=result.final_answer or "",
                save_pdf=save_pdf,
                papers_output_dir=papers_output_dir,
                papers_filename=papers_filename,
                review_output_dir=review_output_dir,
                review_filename=review_filename,
                output_dir=output_dir,
                when=started,
                write_review_pdf=_write_review if save_pdf else None,
            )
            pdf_path = saved.get("pdf_path")
            output_folder = saved.get("query_folder")
            papers_folder = saved.get("papers_folder")
            review_folder = saved.get("review_folder")
            logger.info(
                "输出已保存 | root=%s | lit_source=%s | papers=%d | review=%s",
                output_folder, papers_folder, saved.get("exported_count") or 0, pdf_path or "无",
            )

            with self._lock:
                self._tasks[task_id].update({
                    "status": "done",
                    "final_answer": result.final_answer,
                    "steps": steps_data,
                    "step_count": len(steps_data),
                    "pdf_path": pdf_path,
                    "output_folder": output_folder,
                    "papers_folder": papers_folder,
                    "review_folder": review_folder,
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

    def save_review_pdf(
        self,
        task_id: str,
        review_output_dir: str,
        review_filename: str = "",
    ) -> dict[str, Any]:
        """任务完成后按用户所选路径写出综述 PDF。不选路径则不调用本方法。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError("not_found")
            if task.get("status") != "done":
                raise ValueError("任务尚未完成")
            topic = task.get("topic") or ""
            final_answer = task.get("final_answer") or ""
            steps = list(task.get("steps") or [])
        if not final_answer.strip():
            raise ValueError("综述正文为空，无法保存 PDF")

        from utils.review_output import build_review_pdf_path

        output_path = build_review_pdf_path(
            topic=topic,
            final_answer=final_answer,
            steps=steps,
            review_output_dir=review_output_dir,
            review_filename=review_filename,
        )
        pdf_gen = PDFGeneratorTool()
        pdf_result = pdf_gen.run({
            "text": final_answer,
            "output_path": str(output_path),
        })
        if str(pdf_result).startswith("ERROR"):
            raise RuntimeError(pdf_result)

        pdf_path = str(output_path)
        review_folder = str(output_path.parent)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task["pdf_path"] = pdf_path
                task["review_folder"] = review_folder
        logger.info("用户保存综述 PDF | task_id=%s | path=%s", task_id, pdf_path)
        return {
            "pdf_path": pdf_path,
            "review_folder": review_folder,
        }

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
