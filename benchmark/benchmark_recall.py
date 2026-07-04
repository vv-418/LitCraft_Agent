"""
大规模召回率评测：对比 HyDE / MQE / Hybrid / 传统向量检索。
自动使用 Chroma 中所有可用集合，动态构建跨集合检索场景。
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.vector_store import VectorStoreTool
from tools.advanced_retrieval import AdvancedRetrieval
from llm_client import LitCraftAgentsLLM

MERGED = "benchmark_merged"


def get_collections(vs):
    return sorted([c.name for c in vs.client.list_collections()
                   if not c.name.startswith("benchmark_")])


def recall(retrieved, gt, k):
    if not gt:
        return 0.0
    hits = len(set(list(retrieved)[:k]) & gt)
    norm = min(len(gt), k)
    return hits / norm if norm > 0 else 0.0


def run_one(vs, ar, col_name, query, gt, top_k=10):
    out = {}
    for sname, sfn in [
        ("traditional", lambda q: json.loads(vs._search(col_name, q, top_k=top_k, threshold=0.0))),
        ("hyde", lambda q: json.loads(ar.run({"collection_name": col_name, "query": q, "top_k": top_k, "strategy": "hyde", "threshold": 0.0}))),
        ("mqe", lambda q: json.loads(ar.run({"collection_name": col_name, "query": q, "top_k": top_k, "strategy": "mque", "threshold": 0.0}))),
        ("hybrid", lambda q: json.loads(ar.run({"collection_name": col_name, "query": q, "top_k": top_k, "strategy": "hybrid", "threshold": 0.0}))),
    ]:
        t0 = time.perf_counter()
        try:
            raw = sfn(query)
            ids = set()
            for c in raw.get("chunks", []):
                meta = c.get("metadata") or {}
                cid = meta.get("chunk_id", "")
                if cid:
                    ids.add(str(cid))
        except Exception:
            ids = set()
        t = time.perf_counter() - t0
        out[f"{sname}_retrieved"] = len(ids)
        out[f"{sname}_recall@5"] = round(recall(ids, gt, 5), 4)
        out[f"{sname}_recall@10"] = round(recall(ids, gt, 10), 4)
        out[f"{sname}_time"] = round(t, 2)
    return out


def build_merged(vs, cols):
    # 先删除旧的（可能有错误的 chunk_id）
    if MERGED in {c.name for c in vs.client.list_collections()}:
        try:
            vs.client.delete_collection(MERGED)
            print(f"[SETUP] 删除旧合并集，准备重建")
        except:
            pass
    print(f"[SETUP] 构建合并集合 '{MERGED}'...")
    all_chunks = []
    for cn in cols:
        try:
            col = vs.client.get_collection(name=cn)
            data = col.get(include=["documents", "metadatas"])
            for i, (doc, meta) in enumerate(zip(data.get("documents",[]), data.get("metadatas",[]))):
                # metadata.chunk_id 使用带前缀的 ID，便于 recall 计算
                cid = f"{cn}_{i}"
                all_chunks.append({"chunk_id": cid, "content": doc, "source": cn,
                                   "metadata": {"source_collection": cn, "chunk_id": cid}})
        except Exception as e:
            print(f"  ⚠️ {cn}: {e}")
    json.loads(vs._add_chunks(MERGED, all_chunks))
    print(f"  ✅ 合并: {len(all_chunks)} chunks")


def print_result(label, rows):
    print(f"\n  ── {label} ──")
    base = [r.get("traditional_recall@5", 0) for r in rows]
    for sname, slabel in [("traditional","传统向量"),("hyde","HyDE"),("mqe","MQE"),("hybrid","Hybrid")]:
        vals = [r.get(f"{sname}_recall@5", 0) for r in rows]
        avg = sum(vals)/len(vals)
        t = sum(r.get(f"{sname}_time",0) for r in rows)/len(rows)
        rel = ""
        if sname != "traditional" and base:
            d = sum((v-b)/b*100 for v,b in zip(vals,base) if b>0)
            cnt = sum(1 for v,b in zip(vals,base) if b>0)
            if cnt: rel = f" (Δ={d/cnt:+.1f}%)"
        print(f"    {slabel:<10} recall@5={avg:.3f}  time={t:.1f}s{rel}")


def main():
    print("=" * 75)
    print("  🔬 大规模检索策略召回率评测 (双语 HyDE/MQE)")
    print("=" * 75)
    vs = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
    cols = get_collections(vs)
    total = sum(c.count() for c in vs.client.list_collections() if not c.name.startswith("benchmark_"))
    print(f"  {len(cols)} collections, {total} chunks")
    llm = LitCraftAgentsLLM()
    ar = AdvancedRetrieval(vector_store=vs, llm=llm, num_hypotheses=2, num_query_variants=2)
    build_merged(vs, cols)

    # 生成 6 篇代表性论文 × 2 语言 = 12 个查询（控制评测时长）
    targets = cols[:6]
    queries = []
    for cn in targets:
        n = vs.client.get_collection(name=cn).count()
        label = cn.replace("_", " ").title()
        for q, lang in [(f"{label} research", "EN"), (f"{label} 研究", "CN")]:
            queries.append({"query": q, "target": cn, "label": f"{label[:25]}({lang})",
                            "n_chunks": n})

    # 场景 A: 同集合
    print(f"\n{'='*75}\n  场景 A: 同集合检索（基线）")
    rows_a = []
    for q in queries:
        col = vs.client.get_collection(name=q["target"])
        data = col.get(include=["metadatas"])
        gt = set()
        for m in (data.get("metadatas") or []):
            if m and "chunk_id" in m: gt.add(str(m["chunk_id"]))
        res = run_one(vs, ar, q["target"], q["query"], gt)
        print(f"  [{q['label'][:30]}] 传统={res['traditional_recall@5']:.3f} HyDE={res['hyde_recall@5']:.3f} MQE={res['mqe_recall@5']:.3f} Hybrid={res['hybrid_recall@5']:.3f}")
        rows_a.append({**q, **res})

    # 场景 B: 跨集合
    print(f"\n{'='*75}\n  场景 B: 跨集合检索")
    rows_b = []
    for q in queries:
        cn, label = q["target"], q["label"]
        n = q["n_chunks"]
        gt = {f"{cn}_{i}" for i in range(n)}
        res = run_one(vs, ar, MERGED, q["query"], gt)
        print(f"  [{label[:30]}] 传统={res['traditional_recall@5']:.3f} HyDE={res['hyde_recall@5']:.3f} MQE={res['mqe_recall@5']:.3f} Hybrid={res['hybrid_recall@5']:.3f}")
        rows_b.append({**q, **res})

    print(f"\n{'='*75}")
    print_result("场景 A: 同集合", rows_a)
    print_result("场景 B: 跨集合", rows_b)

    # NLP 类 vs CV 类细分
    nlp = [r for r in rows_b if any(k in r["target"] for k in ("attention","bert","gpt","roberta","distilbert","tensor"))]
    cv = [r for r in rows_b if any(k in r["target"] for k in ("residual","detection","deep_residual","2606"))]
    if nlp: print_result("场景 B/NLP类", nlp)
    if cv: print_result("场景 B/CV类", cv)

    out = ROOT / "output" / "benchmark_recall.json"
    json.dump({"scenario_a": rows_a, "scenario_b": rows_b},
              open(str(out),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out}")


if __name__ == "__main__":
    main()
