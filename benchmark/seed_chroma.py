"""批量处理已下载的 PDF，全部入库 Chroma。"""
import json, sys, re, time
from pathlib import Path
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
    # 下面这些已下载但未入库
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
    """清理名称，符合 Chroma 要求（小写字母数字._-）。"""
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name.lower().strip('._-'))
    return name[:63] if len(name) > 63 else name


def main():
    print("=" * 60)
    print("  批量导入已下载 PDF → Chroma")
    print("=" * 60)

    parser = PDFParserTool(storage_path="./storage/papers")
    chunker = TextChunkerTool(chunk_size=512, chunk_overlap=128)
    vs = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")

    existing = {c.name for c in vs.client.list_collections()}
    pdf_dir = Path("./storage/papers")
    imported = 0

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        fname = pdf_file.stem  # without .pdf

        # 匹配预定义 label
        label = None
        for key, val in PDF_NAMES.items():
            if key in fname or fname.startswith(key):
                label = sanitize(val)
                break
        if label is None:
            # 自动生成
            label = sanitize(fname[:60])

        if label in existing:
            continue

        print(f"\n  [{label}]  {pdf_file.name}")
        pdf_path = str(pdf_file)

        # 1. 解析
        try:
            data = json.loads(parser.run({"pdf_path": pdf_path}))
            if isinstance(data, dict) and data.get("status") == "success":
                c = data.get("content", {}) or {}
                text = c.get("full_text", "") if isinstance(c, dict) else str(c)
            else:
                # 可能返回错误，试试另一种提取方式
                if isinstance(data, dict) and "text" in data:
                    text = data["text"]
                else:
                    print(f"    ⏭ 解析失败, 跳过")
                    continue
        except Exception as e:
            print(f"    ⏭ 解析异常: {e}, 跳过")
            continue

        if not text or len(text) < 200:
            print(f"    ⏭ 提取文本太少 ({len(text)} 字符), 跳过")
            continue

        print(f"    提取 {len(text)} 字符")

        # 2. 分块
        try:
            chunks_json = chunker.run({"text": text[:30000], "source": label})
            chunks = json.loads(chunks_json).get("chunks", [])
        except Exception as e:
            print(f"    ⏭ 分块失败: {e}")
            continue

        if not chunks:
            print(f"    ⏭ 无有效分块")
            continue

        print(f"    生成 {len(chunks)} 个块")

        # 3. 入库
        try:
            r = vs.run({"action": "add", "collection_name": label, "chunks": chunks})
            print(f"    ✅ {str(r)[:80]}")
            imported += 1
            existing.add(label)
        except Exception as e:
            print(f"    ❌ 入库失败: {e}")

    total_chunks = 0
    print(f"\n{'='*60}")
    print(f"  Chroma 集合列表 ({imported} 篇新入库):")
    for c in vs.client.list_collections():
        try:
            cnt = c.count()
            total_chunks += cnt
        except:
            cnt = "?"
        print(f"    - {c.name}: {cnt} chunks")
    print(f"  总 chunks: {total_chunks}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
