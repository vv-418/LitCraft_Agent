"""
LitCraft Agent 集中式日志模块。

提供统一的日志记录能力，同时输出到控制台和文件。
模块在导入时立即初始化根 Logger，从而阻止第三方库
（如 scholarly）调用 logging.basicConfig() 创建无关日志文件。

日志文件位置（位于项目根目录下的 logs/ 文件夹，按日期分目录）：
  logs/YYYY-MM-DD/litcraft_YYYY-MM-DD.log  - 当日主日志
  logs/YYYY-MM-DD/tasks/{task_id}.log      - 当日任务专属日志

使用方式：
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("任务开始")
    logger.error("出错了", exc_info=True)
"""

import logging
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 日志存放目录
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

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
_DAY_FORMAT = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_initialized: bool = False
_file_lock_warned: bool = False


def _date_str(when: Optional[datetime] = None) -> str:
    """返回 YYYY-MM-DD 日期字符串。"""
    return (when or datetime.now()).strftime(_DAY_FORMAT)


def get_day_log_dir(when: Optional[datetime] = None) -> Path:
    """返回当日日志目录：logs/YYYY-MM-DD/，不存在则创建。"""
    day_dir = LOG_DIR / _date_str(when)
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "tasks").mkdir(parents=True, exist_ok=True)
    return day_dir


def _ensure_dirs() -> None:
    """确保根日志目录与当日目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    get_day_log_dir()


class DateFolderFileHandler(logging.Handler):
    """按日期写入 logs/YYYY-MM-DD/litcraft_YYYY-MM-DD.log 的 Handler。

    跨午夜时自动切换到新日期目录与文件名。
    单日文件超过 maxBytes 时在同目录内做体积轮转。
    """

    def __init__(
        self,
        basename: str = "litcraft",
        maxBytes: int = 10 * 1024 * 1024,
        backupCount: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.basename = basename
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.encoding = encoding
        self._current_date: Optional[str] = None
        self._file_handler: Optional[RotatingFileHandler] = None
        self._switch_if_needed()

    def _logfile_path(self, day: str) -> Path:
        day_dir = LOG_DIR / day
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "tasks").mkdir(parents=True, exist_ok=True)
        return day_dir / f"{self.basename}_{day}.log"

    def _switch_if_needed(self) -> None:
        today = _date_str()
        if today == self._current_date and self._file_handler is not None:
            return

        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None

        path = self._logfile_path(today)
        # delay=True：等首次写入再打开，减轻 Windows 上多进程抢同一文件的锁
        self._file_handler = RotatingFileHandler(
            path,
            maxBytes=self.maxBytes,
            backupCount=self.backupCount,
            encoding=self.encoding,
            delay=True,
        )
        self._file_handler.setLevel(self.level)
        if self.formatter:
            self._file_handler.setFormatter(self.formatter)
        self._current_date = today

    def setLevel(self, level) -> None:
        super().setLevel(level)
        if self._file_handler is not None:
            self._file_handler.setLevel(level)

    def setFormatter(self, fmt) -> None:
        super().setFormatter(fmt)
        if self._file_handler is not None:
            self._file_handler.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._switch_if_needed()
            assert self._file_handler is not None
            self._file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """文件被占用时只提示一次，避免每条日志都往 stderr 刷完整堆栈。"""
        global _file_lock_warned
        err = sys.exc_info()[1]
        locked = isinstance(err, OSError) and getattr(err, "winerror", None) in (32, 33)
        locked = locked or (isinstance(err, PermissionError))
        if locked:
            if not _file_lock_warned:
                _file_lock_warned = True
                print(
                    "[LOG] 日志文件被占用（多半是第二个后端窗口，或 Cursor 打开了 .log）。"
                    "本次只打控制台，不再刷屏。请只保留一个 Backend，并关掉日志文件标签页。",
                    file=sys.stderr,
                )
            return
        super().handleError(record)

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()


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
    _remove_all_handlers()

    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    main_file = DateFolderFileHandler(
        basename="litcraft",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
    root.addHandler(main_file)

    # 热重载监视日志文件时会刷 DEBUG；第三方 HTTP / PDF 解析同样过吵
    for noisy in (
        "watchfiles", "watchfiles.main", "httpcore", "httpx", "openai._base_client",
        "pdfminer", "pdfminer.pdfinterp", "pdfminer.psparser", "pdfminer.pdfpage",
        "pdfminer.pdfdocument", "pdfminer.converter", "pdfminer.cmapdb",
        "pdfplumber",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
      - 当日主日志 logs/YYYY-MM-DD/litcraft_YYYY-MM-DD.log
      - 当日任务日志 logs/YYYY-MM-DD/tasks/{task_id}.log

    任务文件落在创建该 logger 当天的日期目录中（跨午夜不挪文件夹）。

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

    day_dir = get_day_log_dir()
    task_handler = RotatingFileHandler(
        day_dir / "tasks" / f"{task_id}.log",
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
