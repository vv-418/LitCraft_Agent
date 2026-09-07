"""综述 / 论文 PDF 的保存目录与文件名。

默认布局（未自选路径时）::

    output/{YYYY-MM-DD}/{搜索主题}/
      lit_source/     检索到的论文 PDF（标题 + 网址）
      figures/        对应插图（子目录名与论文文件名一致）
      papers/         最终综述 PDF
        01{主题}_综述_{HH：MM}.pdf
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_INVALID_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Windows 文件名不能含半角冒号，用全角冒号让时刻看起来仍是 21：20
_CLOCK_SEP = "\uFF1A"


def sanitize_topic_for_filename(topic: str, max_len: int = 40) -> str:
    """把主题收成可做文件名的片段。"""
    text = (topic or "").strip()
    text = _INVALID_FS.sub("", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    if not text:
        text = "untitled"
    return text[:max_len]


def sanitize_user_filename(name: str, default: str = "", max_len: int = 80) -> str:
    """用户自定义文件名：去掉非法字符，不强行改成主题格式。"""
    text = (name or "").strip().strip('"')
    text = os.path.basename(text)
    text = _INVALID_FS.sub("", text)
    text = text.strip(" .")
    if not text:
        return default
    return text[:max_len]


def url_filename_slug(url: str, max_len: int = 80) -> str:
    """把 URL 收成文件名可用片段。"""
    text = (url or "").strip()
    text = re.sub(r"^https?://", "", text, flags=re.I)
    text = _INVALID_FS.sub("_", text)
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._")
    text = re.sub(r"\.pdf$", "", text, flags=re.I)
    return (text or "link")[:max_len]


def paper_export_filename(
    title: str,
    url: str,
    prefix: str = "",
    max_len: int = 180,
) -> str:
    """论文 PDF 文件名：可选前缀 + 论文标题 + URL。"""
    name = sanitize_topic_for_filename(title or "paper", max_len=80)
    slug = url_filename_slug(url)
    combined = f"{name}__{slug}" if slug else name
    pref = sanitize_user_filename(prefix)
    if pref:
        combined = f"{pref}_{combined}"
    return combined[:max_len] + ".pdf"


def default_output_root() -> Path:
    raw = (os.getenv("OUTPUT_PATH") or "./output").strip() or "./output"
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def resolve_save_root(user_dir: str = "") -> Path:
    """用户指定目录优先，否则用项目 output。"""
    raw = (user_dir or "").strip().strip('"')
    if raw:
        return Path(raw).expanduser().resolve()
    return default_output_root()


def build_query_folder(
    topic: str,
    user_dir: str = "",
    when: Optional[datetime] = None,
) -> Path:
    """默认布局的根：{output 或自定义根}/{日期}/{主题}/。"""
    day = (when or datetime.now()).strftime("%Y-%m-%d")
    folder = resolve_save_root(user_dir) / day / sanitize_topic_for_filename(topic, max_len=80)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _clock_stamp(when: Optional[datetime] = None) -> str:
    now = when or datetime.now()
    return f"{now.strftime('%H')}{_CLOCK_SEP}{now.strftime('%M')}"


def next_review_seq(folder: Path, topic: str) -> int:
    """同一主题文件夹下，综述次数从 01 自动递增。"""
    safe_topic = sanitize_topic_for_filename(topic, max_len=60)
    pat = re.compile(rf"^(\d+){re.escape(safe_topic)}_综述_")
    max_n = 0
    if folder.is_dir():
        for item in folder.iterdir():
            if not item.is_file():
                continue
            match = pat.match(item.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
    return max_n + 1


def build_review_pdf_filename(
    topic: str,
    folder: Optional[Path] = None,
    custom_name: str = "",
    when: Optional[datetime] = None,
    **_unused: Any,
) -> str:
    """默认：{次数}{主题}_综述_{HH：MM}.pdf；用户填了名字则用用户的。"""
    custom = sanitize_user_filename(custom_name)
    if custom:
        if not custom.lower().endswith(".pdf"):
            custom += ".pdf"
        return custom
    safe_topic = sanitize_topic_for_filename(topic, max_len=60)
    seq = next_review_seq(folder or Path("."), topic)
    return f"{seq:02d}{safe_topic}_综述_{_clock_stamp(when)}.pdf"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def estimate_review_completeness(
    final_answer: str,
    steps: Optional[list[Any]] = None,
) -> int:
    """估算综述完整度 0–100（保留给其它统计用，不再写入文件名）。"""
    if not final_answer or not final_answer.strip():
        return 0

    score = 0
    length = len(final_answer.strip())
    score += min(40, int(length / 3500 * 40))

    headings = len(re.findall(r"(?m)^#{1,3}\s+\S+", final_answer))
    score += min(30, headings * 5)

    if re.search(r"\[\d+\]|\(\d{4}\)|et\s+al\.?", final_answer, re.I):
        score += 15
    elif re.search(r"参考文献|References", final_answer, re.I):
        score += 10

    n_steps = len(steps or [])
    score += min(15, n_steps * 2)

    return int(min(100, max(0, score)))


def resolve_papers_layout(
    topic: str,
    papers_output_dir: str = "",
    output_dir: str = "",
    when: Optional[datetime] = None,
) -> tuple[Path, Path, Path]:
    """返回 (任务根目录, lit_source 目录, figures 目录)。"""
    custom = (papers_output_dir or output_dir or "").strip().strip('"')
    if custom:
        root = Path(custom).expanduser().resolve()
    else:
        root = build_query_folder(topic, "", when=when)
    lit_source = root / "lit_source"
    figures = root / "figures"
    lit_source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return root, lit_source, figures


def resolve_review_dir(
    topic: str,
    review_output_dir: str = "",
    output_dir: str = "",
    when: Optional[datetime] = None,
) -> Path:
    """综述 PDF 目录：自选路径直接用；否则默认 output/日期/主题/papers/。"""
    custom = (review_output_dir or "").strip().strip('"')
    if custom:
        folder = Path(custom).expanduser().resolve()
    elif (output_dir or "").strip():
        folder = Path(output_dir).expanduser().resolve() / "papers"
    else:
        folder = build_query_folder(topic, "", when=when) / "papers"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def build_review_pdf_path(
    topic: str,
    final_answer: str = "",
    steps: Optional[list[Any]] = None,
    output_dir: str | Path = "",
    review_output_dir: str = "",
    review_filename: str = "",
    when: Optional[datetime] = None,
) -> Path:
    """综述 PDF 完整路径。"""
    folder = resolve_review_dir(
        topic,
        review_output_dir=review_output_dir,
        output_dir=str(output_dir or ""),
        when=when,
    )
    name = build_review_pdf_filename(
        topic,
        folder=folder,
        custom_name=review_filename,
        when=when,
        final_answer=final_answer,
        steps=steps,
    )
    return _unique_path(folder / name)


def _observation_papers(observation: Any) -> list[dict]:
    if not observation:
        return []
    raw = observation
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace < 0:
            return []
        text = text[brace:]
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    papers = data.get("papers") or data.get("results") or []
    found = [p for p in papers if isinstance(p, dict)]
    path = str(data.get("path") or data.get("local_path") or "").strip()
    if path:
        found.append({
            "local_path": path,
            "title": data.get("title") or data.get("filename") or "",
            "url": data.get("url") or data.get("pdf_url") or "",
            "pdf_url": data.get("pdf_url") or "",
        })
    return found


def _iter_downloaded_papers(steps: list[Any]) -> list[dict]:
    seen_src: set[str] = set()
    papers: list[dict] = []
    for step in steps or []:
        obs = getattr(step, "observation", None)
        if obs is None and isinstance(step, dict):
            obs = step.get("observation")
        for paper in _observation_papers(obs):
            src = str(paper.get("local_path") or "").strip()
            if not src or src in seen_src or not os.path.isfile(src):
                continue
            seen_src.add(src)
            papers.append(paper)
    return papers


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def _move_file(src: Path, dest: Path) -> Path | None:
    """把文件移到目标名；已在目标位置则原样返回。"""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and _same_path(src, dest):
            return dest
        if dest.exists():
            dest = _unique_path(dest)
        shutil.move(str(src), str(dest))
        return dest
    except OSError:
        return None


def _move_fig_dir(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists() and _same_path(src_dir, dest_dir):
        return
    if not dest_dir.exists():
        try:
            shutil.move(str(src_dir), str(dest_dir))
        except OSError:
            return
        return
    for img in list(src_dir.iterdir()):
        if not img.is_file():
            continue
        target = dest_dir / img.name
        if target.exists():
            continue
        try:
            shutil.move(str(img), str(target))
        except OSError:
            continue
    try:
        src_dir.rmdir()
    except OSError:
        shutil.rmtree(src_dir, ignore_errors=True)


def export_downloaded_papers(
    steps: list[Any],
    dest_dir: Path,
    filename_prefix: str = "",
    figures_dir: Optional[Path] = None,
) -> list[Path]:
    """下载已在 dest_dir 时只重命名；若仍在其它目录则搬过来（不复制双份）。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []

    for paper in _iter_downloaded_papers(steps):
        src = Path(str(paper.get("local_path") or "").strip())
        if not src.is_file():
            continue
        title = paper.get("title") or src.stem
        url = paper.get("pdf_url") or paper.get("url") or ""
        dest = dest_dir / paper_export_filename(str(title), str(url), prefix=filename_prefix)
        if dest.suffix.lower() != src.suffix.lower() and src.suffix.lower() == ".txt":
            dest = dest.with_suffix(".txt")
        old_stem = src.stem
        result = _move_file(src, dest)
        if result is None:
            continue
        moved.append(result)

        if figures_dir is None:
            continue
        figures_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            figures_dir / old_stem,
            src.parent.parent / "figures" / old_stem,
        ]
        seen: set[str] = set()
        for src_fig in candidates:
            key = str(src_fig)
            if key in seen:
                continue
            seen.add(key)
            _move_fig_dir(src_fig, figures_dir / result.stem)
    return moved


def save_run_outputs(
    topic: str,
    steps: list[Any],
    final_answer: str = "",
    save_pdf: bool = True,
    papers_output_dir: str = "",
    papers_filename: str = "",
    review_output_dir: str = "",
    review_filename: str = "",
    output_dir: str = "",
    when: Optional[datetime] = None,
    write_review_pdf=None,
) -> dict[str, Any]:
    """导出论文/插图（已在目标目录则只重命名），并按需写出综述 PDF。"""
    now = when or datetime.now()
    query_folder, lit_source, figures = resolve_papers_layout(
        topic,
        papers_output_dir=papers_output_dir,
        output_dir=output_dir,
        when=now,
    )
    exported = export_downloaded_papers(
        steps,
        lit_source,
        filename_prefix=papers_filename,
        figures_dir=figures,
    )

    pdf_path = None
    review_folder = None
    if save_pdf and final_answer:
        pdf_path_obj = build_review_pdf_path(
            topic=topic,
            final_answer=final_answer,
            steps=steps,
            output_dir=output_dir,
            review_output_dir=review_output_dir,
            review_filename=review_filename,
            when=now,
        )
        review_folder = str(pdf_path_obj.parent)
        if write_review_pdf is not None:
            result = write_review_pdf(str(pdf_path_obj))
            if not str(result).startswith("ERROR"):
                pdf_path = str(pdf_path_obj)
        else:
            pdf_path = str(pdf_path_obj)

    return {
        "query_folder": str(query_folder),
        "papers_folder": str(lit_source),
        "figures_folder": str(figures),
        "review_folder": review_folder,
        "pdf_path": pdf_path,
        "exported_count": len(exported),
    }
