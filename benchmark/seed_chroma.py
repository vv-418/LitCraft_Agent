"""批量处理已下载的 PDF，全部入库 Chroma。"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.pdf_parser import PDFParserTool
from tools.text_chunker import TextChunkerTool
from tools.vector_store import VectorStoreTool

# 已下载的 PDF → Chroma collection name 映射
PDF_NAMES = {
    "1706.03762": "attention_is_all_you_need",
    "1810.04805": "bert_pretraining",
    "1512.03385": "deep_residual_learning",
    "1907.11692": "roberta",
    "1910.01108": "distilbert",
    "1406.2661": "generative_adversarial_nets",
    "gpt2": "gpt2_language_models",
    "bert": "bert_original_paper",
    "transformer": "transformer_original_paper",
    "deep_learning_object_detection": "object_detection_survey",
    "huggingface": "huggingface_transformers",
    "tensor2tensor": "tensor2tensor_nmt",
    "efficient_weather": "weather_recognition",
    "ammus": "ammus_survey",
}


def sanitize(name: str) -> str:
    """清理名称，符合 Chroma 要求（小写字母数字._-，且首尾为字母数字）。"""
    import hashlib

    raw = (name or "").strip()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", raw.lower())
    name = name.strip("._-")
    name = re.sub(r"_+", "_", name)
    # 纯中文/URL 编码名被清空时，用 hash 保证唯一且合法
    if len(name) < 3:
        digest = hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
        name = f"paper_{digest}"
    name = name[:55]
    if name[0] in "._-":
        name = "c" + name[1:]
    if name[-1] in "._-":
        name = name[:-1] + "x"
    return name

def main():
    args = [a for a in sys.argv[1:] if a != "--rebuild"]
    rebuild = "--rebuild" in sys.argv
    if not args:
        print("用法: python benchmark/seed_chroma.py <pdf目录> [--rebuild]")
        print("例如: python benchmark/seed_chroma.py output/2026-09-07/高速公路天气图像识别/lit_source")
        sys.exit(1)
    pdf_dir = Path(args[0])
    if not pdf_dir.is_dir():
        print(f"目录不存在: {pdf_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Batch import PDFs -> Chroma (multilingual bge-m3)")
    print("=" * 60)
    print(f"  pdf_dir: {pdf_dir}")

    figures_dir = pdf_dir.parent / "figures"
    parser = PDFParserTool(storage_path=str(pdf_dir), figures_path=str(figures_dir))
    chunker = TextChunkerTool(chunk_size=512, chunk_overlap=128)
    vs = VectorStoreTool(
        db_path="./storage/chroma",
        local_model_cache="./models",
    )

    existing = {c.name for c in vs.client.list_collections()}
    imported = 0
    skipped = 0

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        if pdf_file.stat().st_size < 1000:
            print(f"\n  skip tiny file: {pdf_file.name}")
            skipped += 1
            continue

        fname = unquote(pdf_file.stem).strip()
        if not fname or fname in {".", ".."}:
            print(f"\n  skip bad name: {pdf_file.name}")
            skipped += 1
            continue

        label = None
        for key, val in PDF_NAMES.items():
            if key in fname or fname.startswith(key):
                label = sanitize(val)
                break
        if label is None:
            label = sanitize(fname[:60])

        has_new = label in existing
        has_legacy = f"{label}_en" in existing or f"{label}_zh" in existing
        # 无 --rebuild：仅当新多语集合已存在才跳过；只剩旧 _en/_zh 则必须重建
        if has_new and not rebuild:
            skipped += 1
            continue
        if (has_new or has_legacy) and rebuild:
            print(json.loads(vs._delete_collection(label)))
            existing.discard(label)
            existing.discard(f"{label}_en")
            existing.discard(f"{label}_zh")
            existing.discard(f"{label}_mm")
        elif has_legacy and not has_new:
            print(f"\n  [{label}] 检测到旧双语集合，自动重建为 bge-m3 单集合")
            print(json.loads(vs._delete_collection(label)))
            existing.discard(f"{label}_en")
            existing.discard(f"{label}_zh")
            existing.discard(f"{label}_mm")

        print(f"\n  [{label}]  {pdf_file.name[:80]}")
        pdf_path = str(pdf_file)

        try:
            data = json.loads(parser.run({"pdf_path": pdf_path}))
            if isinstance(data, dict) and data.get("status") == "success":
                c = data.get("content", {}) or {}
                text = c.get("full_text", "") if isinstance(c, dict) else str(c)
            elif isinstance(data, dict) and "text" in data:
                text = data["text"]
            else:
                print("    skip: parse failed")
                skipped += 1
                continue
        except Exception as e:
            print(f"    skip: parse error: {e}")
            skipped += 1
            continue

        if not text or len(text) < 200:
            print(f"    skip: too little text ({len(text or '')})")
            skipped += 1
            continue

        print(f"    chars={len(text)}")

        try:
            chunks_json = chunker.run({"text": text[:30000], "source": label})
            chunks = json.loads(chunks_json).get("chunks", [])
        except Exception as e:
            print(f"    skip: chunk error: {e}")
            skipped += 1
            continue

        if not chunks:
            print("    skip: no chunks")
            skipped += 1
            continue

        print(f"    chunks={len(chunks)}")

        try:
            r = vs.run({"action": "add", "collection_name": label, "chunks": chunks})
            print(f"    OK {str(r)[:100]}")
            imported += 1
            existing.add(label)
        except Exception as e:
            print(f"    FAIL: {e}")
            skipped += 1

    total_chunks = 0
    print(f"\n{'=' * 60}")
    print(f"  Imported={imported} skipped={skipped}")
    for c in vs.client.list_collections():
        try:
            cnt = c.count()
            total_chunks += cnt
        except Exception:
            cnt = "?"
        print(f"    - {c.name}: {cnt} chunks")
    print(f"  total chunks: {total_chunks}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
