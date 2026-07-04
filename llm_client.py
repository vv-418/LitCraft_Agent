import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量（override=True 确保覆盖系统环境变量残留）
load_dotenv(override=True)


class LitCraftAgentsLLM:
    """
    LitCraft Agent 的 LLM 客户端。
    用于调用任何兼容 OpenAI 接口的服务，并默认使用流式响应。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        """
        便捷方法：根据系统提示和用户提示调用大模型。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.think(messages, temperature)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        API 出错时自动重试最多 3 次，指数退避。
        """
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[LLM] 正在调用 {self.model} 模型 (第 {attempt}/{max_retries} 次)...", flush=True)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )

                print("[OK] 大语言模型响应成功:", flush=True)
                collected_content = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                print()  # 在流式输出结束后换行
                return "".join(collected_content)

            except Exception as e:
                last_error = e
                err_msg = str(e)[:120]
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2, 4, 8 秒
                    print(f"\n[LLM] ⚠ API 调用失败 (尝试 {attempt}/{max_retries}): {err_msg}", flush=True)
                    print(f"[LLM] 等待 {wait}s 后重试...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"\n[LLM] ❌ API 调用已全部失败 ({max_retries}/{max_retries}): {err_msg}", flush=True)

        logger = __import__('logging').getLogger('llm')
        logger.error("LLM API 调用失败 | model=%s | error=%s", self.model, str(last_error), exc_info=True)
        raise last_error  # type: ignore


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = LitCraftAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
