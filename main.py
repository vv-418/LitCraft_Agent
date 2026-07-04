# 作用：命令行入口脚本，负责创建 Agent、接收研究主题、运行 Agent 并打印结果。
import os
# 强制离线模式：模型已缓存到本地，禁止 huggingface_hub 联网检查更新
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import io
# 设置 stdout 为 UTF-8 编码，避免 Windows GBK 终端下 emoji 报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='backslashreplace')

import argparse

from utils.logger import get_logger

from agent.langgraph_agent import LangGraphAgent
from llm_client import LitCraftAgentsLLM
from tools.registry import ToolRegistry
from tools.search import ArxivSearchTool, SemanticScholarTool, GoogleScholarTool, MultiSourceSearchTool
from tools.paper_downloader import PaperDownloaderTool
from tools.pdf_parser import PDFParserTool
from tools.text_chunker import TextChunkerTool
from tools.vector_store import VectorStoreTool
from tools.advanced_retrieval import AdvancedRetrieval

logger = get_logger("main")


# 作用：创建大模型客户端、工具注册表，并组装成一个可以运行的 LangGraphAgent。
def build_agent() -> LangGraphAgent:
    """创建 LLM、注册工具，并组装成一个 LangGraphAgent。"""
    llm = LitCraftAgentsLLM()

    tools = ToolRegistry()
    
    # 第0步：多源搜索（arXiv + Semantic Scholar + Google Scholar）
    tools.register(MultiSourceSearchTool(max_results=10), aliases=["search", "search_papers"])
    
    # 第1步：搜索论文（arXiv）
    tools.register(ArxivSearchTool(max_results=5, initial_delay=5.0, max_retries=5), aliases=["search_arxiv"])
    
    # 第1b步：Semantic Scholar（补充）
    tools.register(SemanticScholarTool(max_results=5), aliases=["search_semantic_scholar"])
    
    # 第1c步：Google Scholar（备选）
    tools.register(GoogleScholarTool(max_results=5), aliases=["search_google_scholar"])
    
    # 第2步：下载论文（PDF）
    tools.register(PaperDownloaderTool(storage_path="./storage/papers", max_retries=3, timeout=30),
                   aliases=["download_pdf", "download_paper", "download_papers", "download", "fetch_paper"])
    
    # 第3步：解析论文（提取文本）
    tools.register(PDFParserTool(storage_path="./storage/papers"), aliases=["parse_pdf", "pdf_extract"])
    
    # 第4步：分块文本（用于向量化）
    tools.register(TextChunkerTool(chunk_size=512, chunk_overlap=128), aliases=["chunk_text", "split_text"])
    
    # 第5步：存储向量（构建向量数据库）
    vector_store = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
    tools.register(vector_store, aliases=["vector_search", "store_vector", "chroma_search"])
    
    # 第6步：高级检索（HyDE + MQE 混合检索）
    tools.register(AdvancedRetrieval(
        vector_store=vector_store,
        llm=llm,
        num_hypotheses=3,
        num_query_variants=3
    ), aliases=["hybrid_search", "retrieve", "advanced_search", "rag_search"])

    # 创建 Agent：充足的步骤数支持完整工作流（搜索→下载→解析→分块→存储→检索→综述）
    # 包括双语搜索、多篇论文下载解析等，需要足够步数
    return LangGraphAgent(llm=llm, tools=tools, max_steps=20)


# 作用：读取命令行参数，启动 Agent，并把执行轨迹和最终答案打印到终端。
def main() -> None:
    """命令行入口：读取研究主题，运行 Agent，并打印执行过程和最终答案。"""
    parser = argparse.ArgumentParser(
        description="LitCraft：智能文献综述 Agent",
        epilog="示例：python main.py --topic 'Transformer 在自然语言处理中的应用'"
    )
    parser.add_argument("--topic", required=True, help="研究主题（例：Transformer 在 NLP 中的应用）")
    parser.add_argument("--year-from", type=str, default="",
                        help="年份过滤条件，只搜索该年份之后的文献（如 2020）")
    parser.add_argument("--save-pdf", type=str, default="",
                        help="将最终答案保存为 PDF 文件的路径（如 output/review.pdf）")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("[INFO] LitCraft Agent 启动")
    print("="*80)
    print(f"\n[TOPIC] 研究主题：{args.topic}")
    logger.info("Agent 启动 | topic=%s | year_from=%s | save_pdf=%s",
                args.topic, args.year_from, args.save_pdf)
    print("\n[WORKFLOW] 完整工作流：")
    print("  [0] 多源搜索（arxiv_search + semantic_scholar + google_scholar）")
    print("  [1] 下载 PDF（paper_downloader）")
    print("  [2] 解析内容（pdf_parser）")
    print("  [3] 文本分块（text_chunker）")
    print("  [4] 向量存储（vector_store）")
    print("  [5] 智能检索（advanced_search with HyDE+MQE）")
    print("  [6] 生成综述（final_answer）")
    print("\n" + "="*80 + "\n")

    agent = build_agent()
    logger.info("Agent 构建完成，开始执行")

    try:
        result = agent.run(args.topic, year_from=args.year_from)
        logger.info("Agent 执行完毕 | steps=%d | final_answer_len=%d",
                    len(result.steps), len(result.final_answer or ""))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("Agent 执行失败 | error=%s\\n%s", str(e), tb)
        print(f"\n[ERROR] Agent 执行失败: {e}")
        sys.exit(1)
    
    # 如果指定了 --save-pdf，将最终答案保存为 PDF
    if args.save_pdf:
        from tools.pdf_generator import PDFGeneratorTool
        pdf_gen = PDFGeneratorTool()
        pdf_result = pdf_gen.run({"text": result.final_answer, "output_path": args.save_pdf})
        if pdf_result.startswith("ERROR"):
            print(f"\n[ERROR] PDF 保存失败: {pdf_result}")
        else:
            print(f"\n[PDF] 最终答案已保存至: {args.save_pdf}")

    print("\n" + "="*80)
    print("[STEPS] 执行步骤详情")
    print("="*80)
    for index, step in enumerate(result.steps, start=1):
        print(f"\n【步骤 {index}】")
        print(f"[THINK] 思考：{step.thought}")
        if step.action:
            print(f"[ACTION] 行动：{step.action}")
            # 美化 action_input 输出
            import json
            if step.action_input:
                try:
                    action_input_str = json.dumps(step.action_input, ensure_ascii=False, indent=2)
                    print(f"[INPUT] 输入：{action_input_str}")
                except:
                    print(f"[INPUT] 输入：{step.action_input}")
        if step.observation:
            # 截断长的 observation
            obs = str(step.observation)
            if len(obs) > 300:
                print(f"[OUTPUT] 结果：{obs[:300]}... [已截断，共 {len(obs)} 字符]")
            else:
                print(f"[OUTPUT] 结果：{obs}")

    print("\n" + "="*80)
    print("[RESULT] 最终答案")
    print("="*80)
    print(result.final_answer)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
