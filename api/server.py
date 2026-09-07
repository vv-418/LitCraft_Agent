"""
作用：FastAPI 后端服务入口。

提供 RESTful API 接口，供 Vue 前端（或其他客户端）调用：
- POST /api/agent/start     → 启动新任务
- GET  /api/agent/status/{id} → 查询任务状态
- GET  /api/agent/result/{id} → 获取最终结果
- GET  /api/agent/steps/{id}  → 获取执行轨迹
- GET  /api/output/{filename} → 下载 PDF 文件
- GET  /api/history            → 历史任务列表
"""

import os
import sys

# 强制离线模式：模型已缓存到本地，禁止 huggingface_hub 联网检查更新
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# 确保项目根目录在 sys.path 中
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    AgentStartRequest,
    TaskStatusResponse,
    AgentResultResponse,
    StepResponse,
    TaskListItem,
    PickSavePathRequest,
    AgentSavePdfRequest,
)
from api.task_manager import TaskManager
from utils.logger import get_logger

logger = get_logger("api")

# 创建全局任务管理器实例
task_manager = TaskManager(max_workers=2)

# 创建 FastAPI 应用
app = FastAPI(
    title="LitCraft Agent API",
    description="智能文献综述 Agent 的后端服务，支持异步执行和进度轮询。",
    version="1.0.0",
)

# 配置 CORS（允许 Vue 或其他前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """服务根路径：返回基本信息。"""
    return {
        "service": "LitCraft Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/api/agent/start", response_model=TaskStatusResponse)
async def start_agent(req: AgentStartRequest):
    """启动一个新的文献综述 Agent 任务（异步执行）。
    
    请求体示例：
    ```json
    {
        "topic": "Transformer 在自然语言处理中的应用",
        "year_from": "2020",
        "save_pdf": true,
        "per_source_limit": 10,
        "final_limit": 10,
        "papers_output_dir": "",
        "review_output_dir": ""
    }
    ```
    """
    logger.info(
        "收到新任务请求 | topic=%s | year_from=%s | save_pdf=%s | per_source=%s | final=%s",
        req.topic, req.year_from, req.save_pdf, req.per_source_limit, req.final_limit,
    )
    task_id = task_manager.start_task(
        topic=req.topic,
        year_from=req.year_from,
        save_pdf=req.save_pdf,
        per_source_limit=req.per_source_limit,
        final_limit=req.final_limit,
        output_dir=req.output_dir,
        papers_output_dir=req.papers_output_dir,
        papers_filename=req.papers_filename,
        review_output_dir=req.review_output_dir,
        review_filename=req.review_filename,
    )
    task = task_manager.get_result(task_id)
    logger.info("任务已创建 | task_id=%s | topic=%.40s", task_id, req.topic)
    return TaskStatusResponse(
        task_id=task_id,
        status="running",
        created_at=task["created_at"] if task else "",
    )


@app.get("/api/agent/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(task_id: str):
    """轮询指定任务的当前状态。"""
    status = task_manager.get_status(task_id)
    if status is None:
        logger.warning("任务状态查询失败 | task_id=%s | 未找到", task_id)
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    logger.debug("任务状态 | task_id=%s | status=%s", task_id, status)
    task = task_manager.get_result(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        created_at=task["created_at"] if task else "",
    )


@app.get("/api/agent/result/{task_id}", response_model=AgentResultResponse)
async def get_result(task_id: str):
    """获取任务的最终结果（仅在 status=done 时包含 final_answer 和 steps）。"""
    task = task_manager.get_result(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    steps_data = [
        StepResponse(
            thought=s.get("thought", ""),
            action=s.get("action"),
            action_input=s.get("action_input", {}),
            observation=s.get("observation"),
            summary=s.get("summary") or {},
        )
        for s in task.get("steps", [])
    ]

    return AgentResultResponse(
        task_id=task_id,
        topic=task.get("topic", ""),
        status=task["status"],
        final_answer=task.get("final_answer", ""),
        steps=steps_data,
        pdf_path=task.get("pdf_path"),
        output_folder=task.get("output_folder"),
        papers_folder=task.get("papers_folder"),
        review_folder=task.get("review_folder"),
        error=task.get("error"),
        traceback=task.get("traceback"),
        created_at=task.get("created_at", ""),
    )


@app.get("/api/agent/steps/{task_id}")
async def get_steps(task_id: str):
    """获取任务的执行步骤列表（可实时调用，任务未完成时也能返回部分步骤）。"""
    status = task_manager.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    steps = task_manager.get_steps(task_id)
    return {
        "task_id": task_id,
        "status": status,
        "steps": steps,
    }


@app.post("/api/agent/save-pdf/{task_id}")
async def save_task_pdf(task_id: str, req: AgentSavePdfRequest):
    """任务完成后由用户手动保存综述 PDF（先选位置再写文件）。"""
    task = task_manager.get_result(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    try:
        saved = task_manager.save_review_pdf(
            task_id,
            review_output_dir=req.review_output_dir,
            review_filename=req.review_filename,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return saved


@app.get("/api/agent/pdf/{task_id}")
async def download_task_pdf(task_id: str):
    """按任务下载综述 PDF（支持日期/主题子目录以及自定义保存路径）。"""
    task = task_manager.get_result(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    pdf_path = str(task.get("pdf_path") or "").strip()
    if not pdf_path or not os.path.isfile(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, filename=os.path.basename(pdf_path))


@app.post("/api/pick-directory")
async def pick_directory():
    """弹出本机文件夹选择框（仅本地后端有效）。"""
    import asyncio

    def _ask() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            chosen = filedialog.askdirectory(title="选择保存文件夹")
            root.destroy()
            return chosen or ""
        except Exception:
            return ""

    path = await asyncio.to_thread(_ask)
    return {"path": path}


@app.post("/api/pick-save-path")
async def pick_save_path(req: PickSavePathRequest):
    """弹出系统「另存为」对话框：同时选择文件夹并填写文件名。"""
    import asyncio
    from pathlib import Path

    from utils.review_output import sanitize_topic_for_filename, sanitize_user_filename

    kind = (req.kind or "review").strip().lower()
    topic = (req.topic or "").strip()
    initial = sanitize_user_filename(req.initial_file)

    if not initial:
        if kind == "papers":
            initial = "标题__网址.pdf"
        else:
            stem = sanitize_topic_for_filename(topic, max_len=40) if topic else "综述"
            initial = f"{stem}_综述.pdf"
    if not initial.lower().endswith(".pdf"):
        initial += ".pdf"

    def _ask() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            title = "保存下载的论文" if kind == "papers" else "保存文献综述 PDF"
            chosen = filedialog.asksaveasfilename(
                title=title,
                defaultextension=".pdf",
                filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")],
                initialfile=initial,
                confirmoverwrite=kind != "papers",
            )
            root.destroy()
            return chosen or ""
        except Exception:
            return ""

    chosen = await asyncio.to_thread(_ask)
    if not chosen:
        return {"path": "", "directory": "", "filename": ""}

    path = Path(chosen)
    filename = path.name
    stem = path.stem
    # 论文对话框预填的「标题__网址」表示沿用默认命名，只采用所选文件夹
    if kind == "papers" and stem in {"标题__网址", "untitled", "未命名"}:
        filename = ""
    return {
        "path": str(path if filename else path.parent),
        "directory": str(path.parent),
        "filename": filename,
    }


@app.get("/api/output/{filename:path}")
async def download_output(filename: str):
    """下载生成的 PDF 文件。
    
    Args:
        filename: output 目录下的文件名（如 review_abc12345_xxx.pdf）
    """
    output_dir = os.path.join(_project_root, "output")
    file_path = os.path.abspath(os.path.join(output_dir, filename))

    # 安全检查：防止路径穿越访问 output 外的文件
    if not file_path.startswith(os.path.abspath(output_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(file_path, filename=filename)


@app.get("/api/history", response_model=list[TaskListItem])
async def list_history():
    """获取所有历史任务的摘要列表（按创建时间倒序）。"""
    tasks = task_manager.list_tasks()
    return [
        TaskListItem(
            task_id=t["task_id"],
            topic=t["topic"],
            status=t["status"],
            created_at=t["created_at"],
        )
        for t in tasks
    ]


# 作用：直接运行此脚本时启动 Uvicorn 服务器。
if __name__ == "__main__":
    import uvicorn

    # 只监视源码目录：写入 logs/ 不应触发热重载（否则会无限重启刷屏）
    _reload_dirs = [
        os.path.join(_project_root, name)
        for name in ("api", "agent", "tools", "utils")
        if os.path.isdir(os.path.join(_project_root, name))
    ]
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=_reload_dirs,
        reload_excludes=["*.log", "*.pdf", "*.parquet"],
    )
