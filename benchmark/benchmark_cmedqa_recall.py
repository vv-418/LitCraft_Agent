"""
CmedqaRetrieval 中文检索召回率评测（同一 embedding：bge-m3）。

对比检索策略：
  - dense：AdvancedRetrieval 默认 Dense Top-K
  - hybrid：Dense+BM25 → RRF → bge-reranker-v2-m3

数据目录（mteb 结构）：
  datasets/cmedqa/corpus/*.parquet
  datasets/cmedqa/queries/*.parquet
  datasets/cmedqa/data/*.parquet   # qrels: query-id, corpus-id, score

结果：benchmark/benchmark_result/benchmark_cmedqa_recall.json

用法：
  python benchmark/benchmark_cmedqa_recall.py --max-queries 30
  python benchmark/benchmark_cmedqa_recall.py --max-queries 30 --full-corpus
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq

from benchmark.paths import RESULT_DIR, result_path
from tools.advanced_retrieval import AdvancedRetrieval
from tools.vector_store import VectorStoreTool

CMEDQA_DIR = ROOT / "datasets" / "cmedqa"
COLLECTION = "cmedqa_beir"
DB_PATH = "./storage/chroma_cmedqa"


def _first_parquet(folder: Path) -> Path:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 parquet: {folder}")
    return files[0]


def load_corpus() -> list[dict]:
    table = pq.read_table(_first_parquet(CMEDQA_DIR / "corpus"))
    data = table.to_pydict()
    rows = []
    ids = data.get("_id") or data.get("id")
    texts = data["text"]
    titles = data.get("title") or [""] * len(texts)
    for i, doc_id in enumerate(ids):
        title = titles[i] or ""
        text = texts[i] or ""
        content = f"{title}\n{text}".strip() if title else text
        rows.append({"_id": str(doc_id), "content": content})
    return rows


def load_queries() -> dict[str, str]:
    table = pq.read_table(_first_parquet(CMEDQA_DIR / "queries"))
    data = table.to_pydict()
    ids = data.get("_id") or data.get("id")
    return {str(i): (t or "") for i, t in zip(ids, data["text"])}


def load_qrels() -> dict[str, set[str]]:
    table = pq.read_table(_first_parquet(CMEDQA_DIR / "data"))
    data = table.to_pydict()
    qrels: dict[str, set[str]] = defaultdict(set)
    for qid, did, score in zip(data["query-id"], data["corpus-id"], data["score"]):
        try:
            if float(score) > 0:
                qrels[str(qid)].add(str(did))
        except (TypeError, ValueError):
            continue
    return dict(qrels)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(retrieved[:k]) & gold)
    return hits / min(len(gold), k)


def extract_doc_ids(raw: dict) -> list[str]:
    ids: list[str] = []
    seen = set()
    for c in raw.get("chunks", []):
        meta = c.get("metadata") or {}
        cid = str(meta.get("chunk_id") or meta.get("doc_id") or c.get("chunk_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def ensure_index(
    vs: VectorStoreTool,
    docs: list[dict],
    rebuild: bool = False,
) -> int:
    existing = {c.name for c in vs.client.list_collections()}
    primary = COLLECTION if COLLECTION in existing else None
    if primary is None:
        for suffix in ("_zh", "_en"):
            name = f"{COLLECTION}{suffix}"
            if name in existing:
                # 旧双语集合与 bge-m3 不兼容，强制重建
                print(f"[INDEX] 发现旧集合 {name}，将重建为多语单集合 {COLLECTION}")
                rebuild = True
                break

    if primary and not rebuild:
        count = vs.client.get_collection(primary).count()
        if count >= len(docs) * 0.95:
            print(f"[INDEX] 复用集合 {primary}（{count} docs）")
            return count
        print(f"[INDEX] 数量不足（{count}/{len(docs)}），重建")
        rebuild = True

    if rebuild:
        print(json.loads(vs._delete_collection(COLLECTION)))

    print(f"[INDEX] 写入 {len(docs)} 篇到 {COLLECTION}（multilingual bge-m3）...")
    batch = 32  # bge-m3 吃内存，批次略小
    for start in range(0, len(docs), batch):
        part = docs[start : start + batch]
        chunks = [
            {
                "chunk_id": d["_id"],
                "content": d["content"],
                "source": "cmedqa",
                "metadata": {"chunk_id": d["_id"], "doc_id": d["_id"]},
            }
            for d in part
        ]
        result = json.loads(vs._add_chunks(COLLECTION, chunks))
        if result.get("status") != "success":
            raise RuntimeError(result)
        done = min(start + batch, len(docs))
        if done % 256 == 0 or done == len(docs):
            print(f"  ... {done}/{len(docs)}", flush=True)

    # 取多语单集合计数
    existing = {c.name for c in vs.client.list_collections()}
    primary = COLLECTION if COLLECTION in existing else None
    if primary is None:
        raise RuntimeError(f"索引写入后未找到集合 {COLLECTION}")
    count = vs.client.get_collection(primary).count()
    print(f"[INDEX] 完成：{primary} = {count}")
    return count


def select_corpus(
    all_docs: list[dict],
    qrels: dict[str, set[str]],
    eval_qids: list[str],
    full_corpus: bool,
    distractors: int,
    seed: int,
) -> list[dict]:
    if full_corpus:
        return all_docs

    need: set[str] = set()
    for qid in eval_qids:
        need |= qrels.get(qid, set())

    by_id = {d["_id"]: d for d in all_docs}
    selected = [by_id[i] for i in need if i in by_id]

    rng = random.Random(seed)
    others = [d for d in all_docs if d["_id"] not in need]
    rng.shuffle(others)
    selected.extend(others[:distractors])
    print(
        f"[CORPUS] 精简库：gold={len(need)} + distractors={min(distractors, len(others))} "
        f"= {len(selected)}（加 --full-corpus 可用全量 10 万篇）"
    )
    return selected


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
            print(f"  [WARN] {mode}: {e}")
            ids = []
        out[mode] = {"ids": ids, "time": round(time.perf_counter() - t0, 3)}
    return out


def main():
    parser = argparse.ArgumentParser(description="Cmedqa 中文检索召回评测")
    parser.add_argument("--max-queries", type=int, default=30, help="评测查询条数（0=全部）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-corpus", action="store_true", help="索引全部约 10 万文档（很慢）")
    parser.add_argument("--distractors", type=int, default=15000, help="精简库负样本数")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("  CmedqaRetrieval 中文召回评测（bge-m3）")
    print("  dense  vs  rerank（Dense pool + Cross-Encoder）")
    print("=" * 72)

    if not (CMEDQA_DIR / "corpus").exists():
        raise FileNotFoundError(f"缺少数据目录: {CMEDQA_DIR}")

    qrels = load_qrels()
    queries = load_queries()
    eval_qids = [qid for qid in qrels if qid in queries and queries[qid].strip()]
    # 稳定顺序
    eval_qids = sorted(eval_qids)
    if args.max_queries > 0:
        eval_qids = eval_qids[: args.max_queries]

    print(f"  queries={len(eval_qids)}  top_k={args.top_k}")
    print(f"  结果目录: {RESULT_DIR}")

    all_docs = load_corpus()
    docs = select_corpus(
        all_docs, qrels, eval_qids, args.full_corpus, args.distractors, args.seed
    )

    vs = VectorStoreTool(
        db_path=DB_PATH,
        local_model_cache="./models",
    )
    ensure_index(vs, docs, rebuild=args.rebuild_index)
    ar = AdvancedRetrieval(vector_store=vs, topic_threshold=0.0)

    rows = []
    for i, qid in enumerate(eval_qids, 1):
        query = queries[qid]
        gold = qrels[qid]
        print(f"\n[{i}/{len(eval_qids)}] gold={len(gold)} | {query[:60]}")
        strat = run_strategies(ar, query, args.top_k)
        row = {"query_id": qid, "query": query, "gold_count": len(gold), "gold_ids": sorted(gold)}
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
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    summary = {
        "dataset": "cmedqa",
        "num_queries": len(rows),
        "corpus_size_indexed": len(docs),
        "full_corpus": bool(args.full_corpus),
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
        f"R@10={summary['dense_recall@10']:.4f}  t={summary['dense_time_avg']:.2f}s"
    )
    print(
        f"  rerank   R@5={summary['rerank_recall@5']:.4f}  "
        f"R@10={summary['rerank_recall@10']:.4f}  t={summary['rerank_time_avg']:.2f}s"
    )

    out_path = result_path("benchmark_cmedqa_recall.json")
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[SAVE] {out_path}")


if __name__ == "__main__":
    main()
