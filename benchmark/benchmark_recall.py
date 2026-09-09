"""
SciFact 向量检索召回率评测（同一 embedding：bge-m3）。

对比检索策略：
  - dense：AdvancedRetrieval 默认 Dense Top-K
  - hybrid：Dense+BM25 → RRF → bge-reranker-v2-m3

数据目录：datasets/scifact/
结果输出：benchmark/benchmark_result/benchmark_recall.json

用法：
  python benchmark/benchmark_recall.py
  python benchmark/benchmark_recall.py --max-queries 50
  python benchmark/benchmark_recall.py --rebuild-index
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.paths import RESULT_DIR, result_path
from tools.advanced_retrieval import AdvancedRetrieval
from tools.vector_store import VectorStoreTool

SCIFACT_DIR = ROOT / "datasets" / "scifact"
COLLECTION = "scifact_beir"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, set[str]]:
    """query_id -> set(corpus_id)，仅保留 score>0。"""
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            try:
                if float(score) > 0:
                    qrels[qid].add(did)
            except ValueError:
                continue
    return dict(qrels)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(retrieved[:k]) & gold)
    return hits / min(len(gold), k)


def ensure_scifact_index(vs: VectorStoreTool, rebuild: bool = False) -> int:
    """把 SciFact corpus 写入 Chroma；chunk_id = 论文 _id，便于按 qrels 算召回。"""
    corpus_path = SCIFACT_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"找不到 SciFact corpus: {corpus_path}")

    corpus = load_jsonl(corpus_path)
    existing = {c.name for c in vs.client.list_collections()}

    if COLLECTION in existing and not rebuild:
        count = vs.client.get_collection(COLLECTION).count()
        if count >= len(corpus) * 0.95:
            print(f"[INDEX] 复用已有集合 {COLLECTION}（{count} docs）")
            return count
        print(f"[INDEX] 集合文档数不足（{count}/{len(corpus)}），重建")
        rebuild = True

    if rebuild and COLLECTION in existing:
        vs.client.delete_collection(COLLECTION)
        print(f"[INDEX] 已删除旧集合 {COLLECTION}")

    print(f"[INDEX] 写入 SciFact corpus → {COLLECTION}（{len(corpus)} 篇）...")
    batch_size = 64
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start : start + batch_size]
        chunks = []
        for doc in batch:
            doc_id = str(doc["_id"])
            title = doc.get("title") or ""
            text = doc.get("text") or ""
            content = f"{title}\n\n{text}".strip()
            chunks.append({
                "chunk_id": doc_id,
                "content": content,
                "source": "scifact",
                "metadata": {
                    "chunk_id": doc_id,
                    "doc_id": doc_id,
                    "title": title[:200],
                },
            })
        result = json.loads(vs._add_chunks(COLLECTION, chunks))
        if result.get("status") != "success":
            raise RuntimeError(f"写入失败: {result}")
        done = min(start + batch_size, len(corpus))
        if done % 512 == 0 or done == len(corpus):
            print(f"  ... {done}/{len(corpus)}")

    count = vs.client.get_collection(COLLECTION).count()
    print(f"[INDEX] 完成，共 {count} docs")
    return count


def extract_doc_ids(raw: dict) -> list[str]:
    """从检索结果抽出文档 id（保序去重）。"""
    ids: list[str] = []
    seen = set()
    for c in raw.get("chunks", []):
        meta = c.get("metadata") or {}
        cid = str(meta.get("chunk_id") or meta.get("doc_id") or c.get("chunk_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


STRATEGIES = ("dense", "rerank")


def run_strategies(ar: AdvancedRetrieval, query: str, top_k: int) -> dict:
    """同一工具、同一索引，对比 dense / rerank。"""
    out = {}
    for mode in STRATEGIES:
        os.environ["RETRIEVAL_MODE"] = mode
        t0 = time.perf_counter()
        try:
            raw = json.loads(
                ar.run({
                    "collection_name": COLLECTION,
                    "query": query,
                    "top_k": top_k,
                    "threshold": 0.0,
                })
            )
            ids = extract_doc_ids(raw)
        except Exception as e:
            print(f"  [WARN] {mode} 失败: {e}")
            ids = []
        out[mode] = {
            "ids": ids,
            "time": round(time.perf_counter() - t0, 3),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="SciFact 召回率评测")
    parser.add_argument("--split", default="test", choices=["test", "train"], help="qrels 划分")
    parser.add_argument("--max-queries", type=int, default=0, help="最多评测多少条 query（0=全部）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rebuild-index", action="store_true", help="强制重建 Chroma 索引")
    args = parser.parse_args()

    qrels_path = SCIFACT_DIR / "qrels" / f"{args.split}.tsv"
    queries_path = SCIFACT_DIR / "queries.jsonl"
    if not qrels_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"SciFact 数据不完整，请确认存在:\n  {queries_path}\n  {qrels_path}\n  {SCIFACT_DIR / 'corpus.jsonl'}"
        )

    print("=" * 72)
    print("  SciFact 向量检索召回率评测（bge-m3）")
    print("  dense  vs  rerank（Dense pool + Cross-Encoder）")
    print("=" * 72)

    qrels = load_qrels(qrels_path)
    queries = {str(q["_id"]): q.get("text", "") for q in load_jsonl(queries_path)}
    eval_qids = [qid for qid in qrels.keys() if qid in queries and queries[qid].strip()]
    if args.max_queries > 0:
        eval_qids = eval_qids[: args.max_queries]

    print(f"  split={args.split}  queries={len(eval_qids)}  top_k={args.top_k}")
    print(f"  结果目录: {RESULT_DIR}")

    vs = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
    ensure_scifact_index(vs, rebuild=args.rebuild_index)
    ar = AdvancedRetrieval(vector_store=vs, topic_threshold=0.0)

    rows = []
    for i, qid in enumerate(eval_qids, 1):
        query = queries[qid]
        gold = qrels[qid]
        print(f"\n[{i}/{len(eval_qids)}] qid={qid} | gold={len(gold)} | {query[:70]}")
        strat = run_strategies(ar, query, args.top_k)

        row = {
            "query_id": qid,
            "query": query,
            "gold_count": len(gold),
            "gold_ids": sorted(gold),
        }
        for name, payload in strat.items():
            ids = payload["ids"]
            row[f"{name}_retrieved"] = len(ids)
            row[f"{name}_ids"] = ids
            row[f"{name}_recall@5"] = round(recall_at_k(ids, gold, 5), 4)
            row[f"{name}_recall@10"] = round(recall_at_k(ids, gold, 10), 4)
            row[f"{name}_time"] = payload["time"]
            print(
                f"    {name:<12} R@5={row[f'{name}_recall@5']:.3f}  "
                f"R@10={row[f'{name}_recall@10']:.3f}  t={payload['time']:.2f}s"
            )
        rows.append(row)

    def avg(key: str) -> float:
        if not rows:
            return 0.0
        return round(sum(r[key] for r in rows) / len(rows), 4)

    summary = {
        "dataset": "scifact",
        "split": args.split,
        "num_queries": len(rows),
        "collection": COLLECTION,
        "embedding": "bge-m3",
        "dense_recall@5": avg("dense_recall@5"),
        "dense_recall@10": avg("dense_recall@10"),
        "dense_time_avg": avg("dense_time"),
        "rerank_recall@5": avg("rerank_recall@5"),
        "rerank_recall@10": avg("rerank_recall@10"),
        "rerank_time_avg": avg("rerank_time"),
    }

    print("\n" + "=" * 72)
    print("  汇总")
    print(
        f"  dense    R@5={summary['dense_recall@5']:.4f}  "
        f"R@10={summary['dense_recall@10']:.4f}  "
        f"t={summary['dense_time_avg']:.2f}s"
    )
    print(
        f"  rerank   R@5={summary['rerank_recall@5']:.4f}  "
        f"R@10={summary['rerank_recall@10']:.4f}  "
        f"t={summary['rerank_time_avg']:.2f}s"
    )

    out_path = result_path("benchmark_recall.json")
    payload = {
        "summary": summary,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVE] {out_path}")


if __name__ == "__main__":
    main()
