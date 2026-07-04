"""
LitCraft Agent 集中式日志模块。

提供统一的日志记录能力，同时输出到控制台和文件。
模块在导入时立即初始化根 Logger，从而阻止第三方库
（如 scholarly）调用 logging.basicConfig() 创建无关日志文件。

日志文件位置（位于项目根目录下的 logs/ 文件夹）：
  logs/litcraft.log   - 主日志（DEBUG 级别及以上，所有应用日志）
  logs/tasks/         - 按 task_id 归类的任务执行日志（仅保留最近 10 次）

使用方式：
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("任务开始")
    logger.error("出错了", exc_info=True)
"""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 日志存放目录
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
TASK_LOG_DIR = LOG_DIR / "tasks"

# ---------------------------------------------------------------------------
# 日志格式
# ---------------------------------------------------------------------------
_CONSOLE_FORMAT = (
    "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
)
_FILE_FORMAT = (
    "%(asctime)s [%(levelname)-7s] %(name)s "
    "(%(filename)s:%(lineno)d) [%(threadName)s] %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_initialized: bool = False


def _ensure_dirs() -> None:
    """确保日志目录结构存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _remove_all_handlers() -> None:
    """移除根 Logger 上所有已有 Handler（阻止第三方库 handler 污染）。"""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()


def init_logging() -> None:
    """初始化全局日志系统（仅执行一次）。

    模块导入时自动调用，确保在任何第三方库调用 basicConfig() 之前
    就占住根 Logger 的 Handler 槽位，使后续 basicConfig() 变为空操作。
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    _ensure_dirs()

    root = logging.getLogger()
    # 先清除已有 Handler（如果第三方库的 basicConfig 已先执行）
    _remove_all_handlers()

    root.setLevel(logging.DEBUG)

    # ------ 控制台 Handler（INFO 及以上） ------
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    # ------ 主日志文件 Handler（DEBUG 及以上，轮转 10MB） ------
    main_file = RotatingFileHandler(
        LOG_DIR / "litcraft.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
    root.addHandler(main_file)


def get_logger(name: str) -> logging.Logger:
    """获取一个配置好的 Logger 实例。

    Args:
        name: 通常传入 ``__name__``。

    Returns:
        配置好的 Logger 实例，可直接用于记录日志。
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# 任务级日志追踪工具
# ---------------------------------------------------------------------------

# 缓存 {task_id: logger}，避免反复创建 FileHandler
_task_loggers: dict[str, logging.Logger] = {}


def get_task_logger(task_id: str) -> logging.Logger:
    """获取一个按 task_id 分类的任务日志 Logger。

    该 Logger 会将日志同时写入:
      - 全局日志文件（litcraft.log）
      - tasks/{task_id}.log（该任务专属文件）

    用于在任务执行过程中记录完整轨迹，方便出错后精确定位。

    Args:
        task_id: 任务唯一标识符。

    Returns:
        任务专用的 Logger 实例。
    """
    if task_id in _task_loggers:
        return _task_loggers[task_id]

    logger = logging.getLogger(f"task.{task_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    _ensure_dirs()
    task_handler = RotatingFileHandler(
        TASK_LOG_DIR / f"{task_id}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    task_handler.setLevel(logging.DEBUG)
    task_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
    logger.addHandler(task_handler)

    _task_loggers[task_id] = logger
    return logger


def cleanup_task_logger(task_id: str) -> None:
    """清理任务 Logger 的 Handler，避免文件句柄泄漏。"""
    logger = _task_loggers.pop(task_id, None)
    if logger is not None:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def format_exc_for_log(exc: BaseException, limit: Optional[int] = None) -> str:
    """将异常格式化为适合日志记录的字符串（包含完整堆栈）。"""
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit)
    return "".join(tb_lines)


# ===========================================================================
# 模块导入时立即初始化日志系统
# 关键作用：在 scholarly 等第三方库调用 basicConfig() 之前就占住
# 根 Logger 的 Handler 槽位，后续 basicConfig() 会自动变为空操作，
# 从而阻止 scholar.log 等无关文件的产生。
# ===========================================================================
init_logging()
