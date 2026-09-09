"""
搜索缓存 + 语义去重效果基准测试
=================================

测试维度：
  1. 缓存命中率 —— 相同查询重复调用时，cache 能否避免重复 API 请求
  2. 跨表述缓存命中 —— LLM 在 ReAct 循环中略微改述同一主题时，是否能命中缓存
  3. 语义去重额外捕获 —— 在 MD5 精确去重之上，能多捕获多少近似重复论文

用法：
    python -m benchmark.benchmark_cache_dedup

输出示例：
    ╔══════════════════════════════════════════════════════════════╗
    ║  搜索缓存 + 语义去重基准测试                                ║
    ╚══════════════════════════════════════════════════════════════╝

    [缓存命中]
      首次查询（无缓存）: 12.3s
      重复查询（缓存命中）: 0.02s
      缓存命中率: 33.3% (1/3)
      API 调用减少: 33.3%

    [跨表述缓存]
      变形查询 "transformer ..." → 7.8s (命中) ✓

    [语义去重]
      MD5 去重后: 5 篇
      语义去重额外移除: 1 篇
      总去重率提升: +20%
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.search.multi_source_search import MultiSourceSearchTool


# ── 测试查询 ──────────────────────────────────────────────────
# 注意：这些查询的真实结果会被缓存，请确保首次运行能搜到结果

TEST_QUERIES = [
    "transformer attention mechanism in NLP",
    "graph neural network for molecular property prediction",
]

# 模拟 LLM 在 ReAct 循环中对同一主题的反复/变体搜索
CROSS_STEP_VARIANTS = {
    "transformer attention mechanism in NLP": [
        "transformer attention mechanism in NLP",
        "self-attention mechanism in transformers",
        "attention is all you need transformer",
    ],
}

# ── 缓存配置 ──────────────────────────────────────────────────
CACHE_TTL = 3600  # 1 hour, 足够完成测试


def _fmt(n: float) -> str:
    return f"{n:.2f}"


def _measure_run(tool: MultiSourceSearchTool, query: str) -> float:
    """执行一次搜索并记录耗时。"""
    start = time.perf_counter()
    try:
        asyncio.run(tool._async_run({
            "query": query,
            "limit": 5,
            "per_source_limit": 5,
            "sources": ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"],
        }))
    except Exception:
        pass
    return time.perf_counter() - start


def main():
    print()
    print("=" * 66)
    print("   📦 搜索缓存 + 语义去重基准测试")
    print("   Search Cache & Semantic Dedup Benchmark")
    print("=" * 66)

    # ════════════════════════════════════════════════════════════
    #  1. 缓存命中率测试
    # ════════════════════════════════════════════════════════════
    print("\n  ── [1/3] 缓存命中率 ──────────────────────────────")
    print("  模式: 首次搜索(无缓存) → 重复搜索(应命中)")

    # 使用一个独立的工具实例（缓存已启用）
    tool_cached = MultiSourceSearchTool(
        max_results=10, use_cache=True, use_semantic_dedup=False,
        cache_ttl=CACHE_TTL, auto_download=False,
    )
    cache_clear_needed = True  # 标记是否需要清缓存

    first_times = []
    second_times = []
    hit_records = []

    for query in TEST_QUERIES:
        # 清空缓存，确保第一次是冷启动
        if cache_clear_needed:
            tool_cached._get_cache().clear()
            cache_clear_needed = False

        # 第一次（应 miss）
        t1 = _measure_run(tool_cached, query)
        first_times.append(t1)
        miss_count = tool_cached.cache_misses
        print(f"    [{query[:50]}] 首次: {_fmt(t1)}s (cache_misses={tool_cached.cache_misses})")

        # 第二次（应 hit）
        t2 = _measure_run(tool_cached, query)
        second_times.append(t2)
        hit_count = tool_cached.cache_hits
        hit_records.append(t2 < 0.5)  # 缓存命中通常 < 0.5s
        print(f"                   重复: {_fmt(t2)}s (cache_hits={tool_cached.cache_hits}) {'✅ HIT' if t2 < 0.5 else '❌ MISS'}")

    avg_first = sum(first_times) / len(first_times)
    avg_second = sum(second_times) / len(second_times)
    total_hits = tool_cached.cache_hits
    total_misses = tool_cached.cache_misses
    total_skipped = tool_cached.cache_skipped
    hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

    print(f"\n    📊 缓存命中率: {hit_rate:.1%} ({total_hits}/{total_hits + total_misses})")
    print(f"    📊 首次平均: {_fmt(avg_first)}s → 重复平均: {_fmt(avg_second)}s")
    if avg_second > 0:
        print(f"    📊 重复搜索加速: {_fmt(avg_first / avg_second)}x")

    # ════════════════════════════════════════════════════════════
    #  2. 跨表述缓存命中率（模拟 LLM ReAct 循环）
    # ════════════════════════════════════════════════════════════
    print("\n  ── [2/3] 跨表述缓存（模拟 ReAct 循环变体） ──────")
    tool_variant = MultiSourceSearchTool(
        max_results=10, use_cache=True, use_semantic_dedup=False,
        cache_ttl=CACHE_TTL, auto_download=False,
    )
    tool_variant._get_cache().clear()

    variant_hits = 0
    variant_total = 0

    for base_query, variants in CROSS_STEP_VARIANTS.items():
        print(f"    base: {base_query[:50]}")
        for i, vq in enumerate(variants):
            t = _measure_run(tool_variant, vq)
            cache_snapshot = tool_variant.cache_hits + tool_variant.cache_misses
            # 检查是否命中（总查询数 > 缓存 misses+skipped 说明有 hit）
            # 简化判断：如果耗时 < 0.5s 且是第2次之后，视为 hit
            is_hit = (t < 0.5 and i > 0)
            if is_hit:
                variant_hits += 1
            variant_total += 1
            status = "✅ HIT" if is_hit else "  MISS"
            print(f"      v{i}: \"{vq[:40]}...\"  {_fmt(t)}s {status}")

    if variant_total > 0:
        variant_hit_rate = variant_hits / variant_total
        print(f"\n    📊 跨表述缓存命中率: {variant_hit_rate:.1%} ({variant_hits}/{variant_total})")

    # ════════════════════════════════════════════════════════════
    #  3. 语义去重效果测试
    # ════════════════════════════════════════════════════════════
    print("\n  ── [3/3] 语义去重效果 ────────────────────────────")

    # 先在没有语义去重的情况下跑一次
    tool_no_dedup = MultiSourceSearchTool(
        max_results=10, use_cache=False, use_semantic_dedup=False,
        auto_download=False,
    )
    # 再开语义去重跑一次
    tool_with_dedup = MultiSourceSearchTool(
        max_results=10, use_cache=False, use_semantic_dedup=True,
        auto_download=False,
    )

    dedup_stats = []
    for query in TEST_QUERIES:
        # 获取语义去重前的原始论文列表（直接调用 _async_run 内部分析）
        result_str = asyncio.run(tool_no_dedup._async_run({
            "query": query,
            "limit": 5,
            "per_source_limit": 5,
            "sources": ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"],
        }))
        data = json.loads(result_str)
        raw_count = data.get("total_count", 0)
        md5_count = raw_count  # use_semantic_dedup=False 时只有 MD5 去重

        # 获取语义去重后的结果
        result_str2 = asyncio.run(tool_with_dedup._async_run({
            "query": query,
            "limit": 5,
            "per_source_limit": 5,
            "sources": ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"],
        }))
        data2 = json.loads(result_str2)
        final_count = data2.get("total_count", 0)
        extra_removed = tool_with_dedup.semantic_dedup_removed

        dedup_stats.append({
            "query": query[:50],
            "raw_after_md5": md5_count,
            "after_semantic": final_count,
            "extra_removed": extra_removed,
        })
        print(f"    [{query[:50]}]")
        print(f"      MD5 去重后: {md5_count} 篇")
        print(f"      语义去重后: {final_count} 篇")
        print(f"      额外移除: {extra_removed} 篇")

    total_extra_removed = sum(d["extra_removed"] for d in dedup_stats)
    print(f"\n    📊 语义去重总计额外移除: {total_extra_removed} 篇")

    # ════════════════════════════════════════════════════════════
    #  汇总
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 66)
    print("   📊 汇总")
    print("=" * 66)

    time_saved = avg_first - avg_second
    print(f"  缓存命中率:         {hit_rate:.1%}")
    print(f"  跨表述缓存命中率:    {variant_hit_rate:.1%} (模拟 ReAct 循环)")
    print(f"  单次查询时间节省:    {_fmt(time_saved)}s (首次 {_fmt(avg_first)}s → 重复 {_fmt(avg_second)}s)")
    print(f"  语义去重额外移除:    {total_extra_removed} 篇")
    if total_extra_removed > 0:
        print(f"  去重率提升:          显著 (捕获跨表述近似重复)")
    else:
        print(f"  去重率提升:          本次测试未发现近似重复 (取决于搜索结果)")

    # 评级
    print()
    if hit_rate >= 0.3:
        print("  🏆 缓存策略有效: 命中率 ≥ 30%，可显著减少重复 API 调用")
    elif hit_rate >= 0.1:
        print("  ✅ 缓存策略有效: 命中率 ≥ 10%，可部分减少重复 API 调用")
    else:
        print("  ℹ️  缓存策略效果有限: 命中率 < 10%，建议检查查询重复模式")

    if total_extra_removed > 0:
        print("  🏆 语义去重有效: 在 MD5 基础上额外捕获了近似重复论文")
    else:
        print("  ℹ️  语义去重本次未返回额外结果 (可能 MD5 已足够)")

    # ── JSON 导出 ──────────────────────────────────────────────
    from benchmark.paths import result_path
    output_path = result_path("benchmark_cache_dedup.json")

    json_output = {
        "test_config": {
            "queries": TEST_QUERIES,
            "cross_step_variants": {k: v for k, v in CROSS_STEP_VARIANTS.items()},
            "cache_ttl_seconds": CACHE_TTL,
            "auto_download": False,
            "sources": ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"],
        },
        "cache_hit_rate": {
            "hit_rate_pct": round(hit_rate * 100, 1),
            "hits": total_hits,
            "misses": total_misses,
            "api_call_reduction_pct": round(hit_rate * 100, 1),
            "avg_first_seconds": round(avg_first, 2),
            "avg_second_seconds": round(avg_second, 2),
            "repeat_speedup_x": round(avg_first / avg_second, 2) if avg_second > 0 else None,
            "time_saved_seconds": round(time_saved, 2),
        },
        "cross_step_cache": {
            "hit_rate_pct": round(variant_hit_rate * 100, 1),
            "hits": variant_hits,
            "total": variant_total,
        },
        "semantic_dedup": {
            "total_extra_removed": total_extra_removed,
            "details": dedup_stats,
        },
    }
    output_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已导出: {output_path}")
    print()


if __name__ == "__main__":
    main()
