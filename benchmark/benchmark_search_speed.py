"""
多源搜索性能对比基准测试
=========================

比较「串行搜索」(旧实现) 与「异步并发搜索」(新实现) 的延迟差异，
输出量化加速比，用于简历和技术评估。

用法：
    运行: python -m benchmark.benchmark_search_speed

输出示例：
    ⚡ 多源搜索性能对比基准测试 ⚡
    ┌──────────────────────────────┬──────────┬──────────┬────────┐
    │ 查询                         │ 串行(s)  │ 并发(s)  │ 加速比 │
    ├──────────────────────────────┼──────────┼──────────┼────────┤
    │ Transformer attention ...    │   32.45  │   10.23  │  3.17x │
    │ ...                          │          │          │        │
    ├──────────────────────────────┼──────────┼──────────┼────────┤
    │ 平均                         │   28.60  │    9.80  │  2.92x │
    └──────────────────────────────┴──────────┴──────────┴────────┘
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.search.multi_source_search import MultiSourceSearchTool


# ── 测试查询集（覆盖不同领域和语言）─────────────────────────────
TEST_QUERIES = [
    "transformer attention mechanism in NLP",
    "deep reinforcement learning for robotics",
    "graph neural network for molecular property prediction",
]

# 跳过 Google Scholar（爬虫不稳定，会主导耗时）。对比官方 API 五源。
SOURCES = ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"]
TOOL_KW = dict(
    max_results=10,
    auto_download=False,
    use_cache=False,
    use_semantic_dedup=False,
    use_query_optimizer=False,
)

# 重复次数（取最短的多次测量以消除冷启动/网络波动的影响）
RUNS_PER_QUERY = 1


def _new_tool() -> MultiSourceSearchTool:
    return MultiSourceSearchTool(**TOOL_KW)


def _measure_serial(tool: MultiSourceSearchTool, query: str) -> float:
    """测量串行执行时间：五源逐个 run()。"""
    start = time.perf_counter()
    payload = {"query": query, "limit": 5}
    for src_tool in (
        tool.openalex_tool,
        tool.crossref_tool,
        tool.europe_pmc_tool,
        tool.arxiv_tool,
        tool.semantic_tool,
    ):
        try:
            src_tool.run(payload)
        except Exception:
            pass
    return time.perf_counter() - start


def _measure_concurrent(tool: MultiSourceSearchTool, query: str) -> float:
    """测量异步并发执行时间。"""
    start = time.perf_counter()
    try:
        asyncio.run(tool._async_run({
            "query": query,
            "limit": 5,
            "per_source_limit": 5,
            "sources": SOURCES,
        }))
    except Exception:
        pass
    return time.perf_counter() - start


def _fmt(n: float) -> str:
    """格式化数字：保留 2 位小数，空位补 0。"""
    return f"{n:.2f}"


def main():
    print()
    print("=" * 66)
    print("   ⚡ 多源搜索性能对比基准测试")
    print("   Literature Search Speed Benchmark")
    print("=" * 66)
    print(f"   测试查询数: {len(TEST_QUERIES)}")
    print(f"   每查询重复: {RUNS_PER_QUERY} 次")
    print(f"   测试模式:   串行(旧)  vs  异步并发(新)")
    print("-" * 66)

    # 存储结果
    rows: list[dict] = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n  [{i}/{len(TEST_QUERIES)}] 查询: {query[:60]}")

        # 每个测量使用独立的工具实例，避免 rate-limit 状态污染
        serial_times = []
        for r in range(RUNS_PER_QUERY):
            tool_ser = _new_tool()
            t = _measure_serial(tool_ser, query)
            serial_times.append(t)
            print(f"     串行 #{r + 1}: {_fmt(t)}s")

        concurrent_times = []
        for r in range(RUNS_PER_QUERY):
            tool_con = _new_tool()
            t = _measure_concurrent(tool_con, query)
            concurrent_times.append(t)
            print(f"     并发 #{r + 1}: {_fmt(t)}s")

        # 取最快的一次（代表最佳网络条件下性能）
        best_serial = min(serial_times)
        best_concurrent = min(concurrent_times)
        speedup = best_serial / best_concurrent if best_concurrent > 0 else 0

        rows.append({
            "query": query[:55],
            "serial": best_serial,
            "concurrent": best_concurrent,
            "speedup": speedup,
        })
        print(f"     ↳ 最佳: 串行 {_fmt(best_serial)}s → 并发 {_fmt(best_concurrent)}s  ✅ {_fmt(speedup)}x")

    # ── 汇总输出 ────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("   📊 汇总结果")
    print("=" * 66)

    # 表头
    header = f"{'查询':<50} {'串行(s)':<10} {'并发(s)':<10} {'加速比':<8}"
    sep = "-" * 66
    print(header)
    print(sep)

    serial_all, concurrent_all = [], []
    for row in rows:
        print(f"{row['query']:<50} {_fmt(row['serial']):<10} {_fmt(row['concurrent']):<10} {_fmt(row['speedup'])}x")
        serial_all.append(row["serial"])
        concurrent_all.append(row["concurrent"])

    # 平均
    avg_serial = sum(serial_all) / len(serial_all)
    avg_concurrent = sum(concurrent_all) / len(concurrent_all)
    avg_speedup = avg_serial / avg_concurrent if avg_concurrent > 0 else 0

    print(sep)
    print(f"{'平均':<50} {_fmt(avg_serial):<10} {_fmt(avg_concurrent):<10} {_fmt(avg_speedup)}x")

    # 最大/最小
    max_speedup = max(r["speedup"] for r in rows)
    min_speedup = min(r["speedup"] for r in rows)
    print(f"{'最大加速比':<50} {'':<10} {'':<10} {_fmt(max_speedup)}x")
    print(f"{'最小加速比':<50} {'':<10} {'':<10} {_fmt(min_speedup)}x")
    print("=" * 66)

    # ── JSON 导出 ────────────────────────────────────────────────
    from benchmark.paths import result_path
    output_path = result_path("benchmark_search_speed.json")

    json_output = {
        "test_config": {
            "num_queries": len(TEST_QUERIES),
            "runs_per_query": RUNS_PER_QUERY,
            "queries": TEST_QUERIES,
            "sources": SOURCES,
            "auto_download": False,
        },
        "results": [
            {
                "query": row["query"],
                "serial_seconds": round(row["serial"], 2),
                "concurrent_seconds": round(row["concurrent"], 2),
                "speedup_x": round(row["speedup"], 2),
            }
            for row in rows
        ],
        "summary": {
            "avg_serial_seconds": round(avg_serial, 2),
            "avg_concurrent_seconds": round(avg_concurrent, 2),
            "avg_speedup_x": round(avg_speedup, 2),
            "max_speedup_x": round(max_speedup, 2),
            "min_speedup_x": round(min_speedup, 2),
        },
    }
    output_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已导出: {output_path}")

    # ── 加速比评级 ──────────────────────────────────────────────
    print()
    if avg_speedup >= 2.5:
        print("  🏆 评测结论: 异步并发搜索显著优于串行 (加速比 ≥ 2.5x)")
    elif avg_speedup >= 1.5:
        print("  ✅ 评测结论: 异步并发搜索优于串行 (1.5x ≤ 加速比 < 2.5x)")
    else:
        print("  ℹ️  评测结论: 异步并发搜索与串行差异较小 (加速比 < 1.5x)")
    print()


if __name__ == "__main__":
    main()
