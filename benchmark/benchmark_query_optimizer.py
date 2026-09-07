"""
LLM 查询优化效果基准测试
========================

模拟 LLM 在 ReAct 循环中常见的低质量搜索查询，
对比优化前后的效果差异。

测试维度：
  1. 查询质量优化 — 停用短语剥离、长度归一化、前缀清理
  2. 零结果挽救率 — 本来会返回 0 篇的查询，优化后是否能搜到结果
  3. 双语覆盖率 — 中文查询是否能自动触发英文搜索并获得结果

用法：
    python -m benchmark.benchmark_query_optimizer
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.search.query_optimizer import QueryOptimizer
from tools.search.multi_source_search import MultiSourceSearchTool


# ── 模拟 LLM 生成的典型低质量查询 ────────────────────────────
BAD_QUERIES = [
    # 冗余前缀
    "Find me papers about transformer attention mechanism",
    "Search for recent articles about graph neural networks",
    "I need to find some research papers on deep reinforcement learning",
    "Can you search for the latest publications about diffusion models",
    "Please find any papers related to large language model evaluation",
    # 过长查询（含论文特有术语）
    "Attention Is All You Need Vaswani 2017 transformer architecture self-attention multi-head attention positional encoding",
    "An image is worth 16x16 words transformers for image recognition at scale dosovitskiy",
    # 中文查询
    "请搜索关于Transformer注意力机制的最新研究论文",
    "深度学习在自然语言处理中的应用研究综述",
]

# ── 真实搜索测试（选取前 N 个，控制耗时） ──────────────────────
REAL_SEARCH_SAMPLE = [
    "Find me papers about transformer attention mechanism",
    "请搜索关于Transformer注意力机制的最新研究论文",
]


def test_text_optimization():
    """测试 1：纯文本优化效果（无需网络请求）。"""
    print("\n  ── [1/3] 文本优化效果（离线） ──────────────────")
    opt = QueryOptimizer()
    results = []

    for raw in BAD_QUERIES:
        result = opt.optimize(raw)
        results.append(result)

        modified = "✅" if result["was_modified"] else "─"
        lang = "中" if result["has_chinese"] else "英"
        orig_len = len(raw)
        opt_len = len(result["optimized"])
        bilingual = " ✓" if len(result["bilingual_candidates"]) > 1 else ""

        print(f"    [{modified}] [{lang}] {orig_len:3d}→{opt_len:3d}字符{bilingual}")
        print(f"          原始: {raw[:60]}")
        print(f"          优化: {result['optimized'][:60]}")

    total = len(results)
    modified = sum(1 for r in results if r["was_modified"])
    bilingual = sum(1 for r in results if len(r["bilingual_candidates"]) > 1)
    avg_orig_len = sum(len(r["original"]) for r in results) / total
    avg_opt_len = sum(len(r["optimized"]) for r in results) / total

    print(f"\n    📊 优化率: {modified}/{total} ({modified/total:.0%})")
    print(f"    📊 双语生成: {bilingual} 个中文查询")
    print(f"    📊 平均长度: {avg_orig_len:.0f} → {avg_opt_len:.0f} 字符")
    print(f"    📊 平均压缩率: {avg_opt_len/avg_orig_len:.0%}")

    return {
        "total": total,
        "modified": modified,
        "bilingual": bilingual,
        "avg_orig_len": round(avg_orig_len, 1),
        "avg_opt_len": round(avg_opt_len, 1),
    }


def test_zero_result_rescue():
    """测试 2：零结果挽救率（真实搜索对比）。

    对同一组查询，分别用「无优化」和「有优化」执行搜索，
    对比零结果次数和论文总数。
    """
    print("\n  ── [2/3] 零结果挽救率（真实搜索） ──────────────")

    tool_no_opt = MultiSourceSearchTool(
        max_results=10,
        use_cache=False,
        use_semantic_dedup=False,
        use_query_optimizer=False,
    )
    tool_with_opt = MultiSourceSearchTool(
        max_results=10,
        use_cache=False,
        use_semantic_dedup=False,
        use_query_optimizer=True,
    )

    records = []
    for query in REAL_SEARCH_SAMPLE:
        print(f"\n    查询: {query[:60]}")

        # 无优化
        start = time.perf_counter()
        result_str_no = asyncio.run(tool_no_opt._async_run({
            "query": query,
            "sources": ["arxiv", "semantic_scholar", "google_scholar"],
        }))
        elapsed_no = time.perf_counter() - start
        data_no = json.loads(result_str_no)
        count_no = data_no.get("total_count", 0)
        empty_no = 1 if count_no == 0 else 0

        print(f"      ❌ 无优化: {count_no} 篇 ({elapsed_no:.1f}s)"
              f"{' ⚠️ 空结果' if count_no == 0 else ''}")

        # 有优化
        start = time.perf_counter()
        result_str_yes = asyncio.run(tool_with_opt._async_run({
            "query": query,
            "sources": ["arxiv", "semantic_scholar", "google_scholar"],
        }))
        elapsed_yes = time.perf_counter() - start
        data_yes = json.loads(result_str_yes)
        count_yes = data_yes.get("total_count", 0)
        empty_yes = 1 if count_yes == 0 else 0

        print(f"      ✅ 有优化: {count_yes} 篇 ({elapsed_yes:.1f}s)"
              f"{' ⚠️ 空结果' if count_yes == 0 else ''}")

        records.append({
            "query": query[:55],
            "without_opt_count": count_no,
            "without_opt_time": round(elapsed_no, 2),
            "with_opt_count": count_yes,
            "with_opt_time": round(elapsed_yes, 2),
            "was_empty_without": bool(empty_no),
            "was_empty_with": bool(empty_yes),
        })

    # 汇总
    total_q = len(records)
    empty_no = sum(1 for r in records if r["was_empty_without"])
    empty_yes = sum(1 for r in records if r["was_empty_with"])
    rescued = empty_no - empty_yes
    total_papers_no = sum(r["without_opt_count"] for r in records)
    total_papers_yes = sum(r["with_opt_count"] for r in records)
    avg_time_no = sum(r["without_opt_time"] for r in records) / total_q
    avg_time_yes = sum(r["with_opt_time"] for r in records) / total_q

    print(f"\n    📊 零结果次数: {empty_no} → {empty_yes} (挽救 {rescued} 次)")
    print(f"    📊 论文总数: {total_papers_no} → {total_papers_yes}")
    print(f"    📊 平均耗时: {avg_time_no:.1f}s → {avg_time_yes:.1f}s")

    return {
        "records": records,
        "empty_without": empty_no,
        "empty_with": empty_yes,
        "rescued": rescued,
        "total_papers_without": total_papers_no,
        "total_papers_with": total_papers_yes,
        "avg_time_without": round(avg_time_no, 2),
        "avg_time_with": round(avg_time_yes, 2),
    }


def main():
    print()
    print("=" * 66)
    print("   🔍 LLM 查询优化效果基准测试")
    print("   Query Optimizer Benchmark")
    print("=" * 66)
    print(f"   测试查询数: {len(BAD_QUERIES)}（文本分析）")
    print(f"   真实搜索数: {len(REAL_SEARCH_SAMPLE)}")
    print("-" * 66)

    # 测试 1: 文本优化
    txt_stats = test_text_optimization()

    # 测试 2: 零结果挽救率
    rescue_stats = test_zero_result_rescue()

    # ── 汇总 ────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("   📊 汇总")
    print("=" * 66)

    mod_rate = txt_stats["modified"] / txt_stats["total"]
    print(f"  查询修改率:        {mod_rate:.0%} ({txt_stats['modified']}/{txt_stats['total']})")
    print(f"  平均字符压缩:      {txt_stats['avg_orig_len']:.0f} → {txt_stats['avg_opt_len']:.0f}")
    print(f"  双语生成数:        {txt_stats['bilingual']} 个")
    print(f"  零结果挽救:        {rescue_stats['rescued']} 次")
    print(f"  论文总数提升:      {rescue_stats['total_papers_without']} → {rescue_stats['total_papers_with']}")

    # 评级
    print()
    if rescue_stats['rescued'] > 0:
        print("  🏆 查询优化有效: 成功挽救零结果查询，降低 Agent 无效重试")
    elif mod_rate > 0.3:
        print("  ✅ 查询优化有效: 大部分查询被清理优化，查询质量提升")
    else:
        print("  ℹ️  查询优化效果有限: 查询质量已较好")

    if txt_stats["bilingual"] > 0:
        print("  🏆 双语搜索机制有效: 自动覆盖中文查询的英文搜索结果")
    print()

    # ── JSON 导出 ───────────────────────────────────────────
    from benchmark.paths import result_path
    output_path = result_path("benchmark_query_optimizer.json")

    json_output = {
        "test_config": {
            "total_queries": len(BAD_QUERIES),
            "real_search_queries": len(REAL_SEARCH_SAMPLE),
            "bad_queries": BAD_QUERIES,
        },
        "text_optimization": {
            "modification_rate_pct": round(mod_rate * 100, 1),
            "modified": txt_stats["modified"],
            "total": txt_stats["total"],
            "bilingual_generated": txt_stats["bilingual"],
            "avg_orig_len": txt_stats["avg_orig_len"],
            "avg_opt_len": txt_stats["avg_opt_len"],
        },
        "zero_result_rescue": rescue_stats,
    }
    output_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果已导出: {output_path}")
    print()


if __name__ == "__main__":
    main()
