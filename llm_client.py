import os
import json
import time
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, List, Dict

# 加载 .env 文件中的环境变量（override=True 确保覆盖系统环境变量残留）
load_dotenv(override=True)


def _native_tools_wanted() -> bool:
    return os.getenv("LLM_NATIVE_TOOLS", "1").strip().lower() not in ("0", "false", "no", "off")


def _is_tools_unsupported_error(err: Exception) -> bool:
    text = str(err).lower()
    keys = (
        "tools",
        "tool_choice",
        "tool_calls",
        "unknown parameter",
        "unexpected keyword",
        "does not support",
        "not supported",
        "function calling",
        "invalid parameter",
    )
    return any(k in text for k in keys)


@dataclass
class LLMTurn:
    """一次助手回复：文本 和/或 一次原生 tool_call。"""
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    used_native_tools: bool = False


def parse_assistant_message(message: Any) -> LLMTurn:
    """从 OpenAI-compatible message 抽出文本或第一个 tool_call。"""
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str):
        content = str(content or "")
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls and isinstance(message, dict):
        content = str(message.get("content") or content)
        tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = str((fn or {}).get("name") or "")
            raw_args = (fn or {}).get("arguments") or "{}"
        else:
            name = str(getattr(fn, "name", "") or "")
            raw_args = getattr(fn, "arguments", None) or "{}"
        args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args = parsed
            except (json.JSONDecodeError, TypeError):
                args = {"raw": str(raw_args)}
        return LLMTurn(content=content.strip(), tool_name=name, tool_args=args, used_native_tools=True)
    return LLMTurn(content=content.strip(), used_native_tools=False)

# 加载 .env 文件中的环境变量（override=True 确保覆盖系统环境变量残留）
load_dotenv(override=True)


def resolve_user_llm_kwargs(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    """解析用户覆盖的大模型参数。

    online：三项都填齐才覆盖，否则返回 None（继续用 .env）。
    local：未填项用 .env 补齐；三项都空则返回 None。
    """
    source_name = str(source or "").strip().lower()
    if source_name not in ("local", "online"):
        source_name = "online"
    model_id = str(model or "").strip()
    key = str(api_key or "").strip()
    url = str(base_url or "").strip()
    if source_name == "local":
        if not (model_id or key or url):
            return None
        model_id = model_id or (os.getenv("LLM_MODEL_ID") or "").strip()
        key = key or (os.getenv("LLM_API_KEY") or "").strip()
        url = url or (os.getenv("LLM_BASE_URL") or "").strip()
    if not (model_id and key and url):
        return None
    kwargs: dict[str, Any] = {"model": model_id, "apiKey": key, "baseUrl": url}
    if timeout is not None and str(timeout).strip() != "":
        try:
            seconds = int(timeout)
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            kwargs["timeout"] = seconds
    return kwargs


def public_llm_defaults() -> dict[str, Any]:
    """给前端展示的默认模型信息，不含 API Key。"""
    timeout_raw = os.getenv("LLM_TIMEOUT", "60")
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        timeout = 60
    model = (os.getenv("LLM_MODEL_ID") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "").strip()
    has_key = bool((os.getenv("LLM_API_KEY") or "").strip())
    return {
        "model": model,
        "base_url": base_url,
        "timeout": timeout,
        "ready": bool(model and base_url and has_key),
    }


class LitCraftAgentsLLM:
    """
    LitCraft Agent 的 LLM 客户端。
    用于调用任何兼容 OpenAI 接口的服务，并默认使用流式响应。
    支持动态感知模型上下文窗口大小。
    """

    # 常见模型 → 上下文窗口 token 数映射(兜底默认 4096)
    _MODEL_CONTEXT_LIMITS: dict[str, int] = {
        # OpenAI 系列
        "gpt-4o":              128000,
        "gpt-4o-mini":         128000,
        "gpt-4-turbo":         128000,
        "gpt-4":                8192,
        "gpt-3.5-turbo":        4096,
        "gpt-3.5-turbo-16k":   16384,
        "gpt-4-1106-preview":  128000,
        "gpt-4-vision-preview": 128000,
        "o1":                   128000,
        "o1-mini":              128000,
        "o3-mini":              200000,
        "deepseek-chat":        65536,
        "deepseek-reasoner":    65536,
        # Anthropic
        "claude-3-5-sonnet":    200000,
        "claude-3-haiku":       200000,
        "qwen2.5":             32768,
        "qwen2":               32768,
        "the AI-plus":            32768,
        "the AI-max":             32768,
        "the AI-max-longcontext": 1000000,
        # GLM
        "glm-4":                128000,
        "glm-3-turbo":           8192,
        # Gemini
        "gemini-1.5-pro":      1048576,
        "gemini-1.5-flash":    1048576,
        "gemini-2.0-flash":    1048576,
    }

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数,如果未提供,则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)
        self._tools_unsupported = not _native_tools_wanted()

    @property
    def max_context_tokens(self) -> int:
        """返回当前模型的上下文窗口大小(token数)。

        查找策略:
        1. 精确匹配(如 'gpt-4o')
        2. 子串包含匹配(如 'gpt-4o-2024-05-13' 包含 'gpt-4o')
        3. 兜底默认 4096
        """
        model_lower = self.model.lower()

        # 精确匹配
        if model_lower in self._MODEL_CONTEXT_LIMITS:
            return self._MODEL_CONTEXT_LIMITS[model_lower]

        # 子串包含匹配(按 key 长度降序,优先匹配最具体的)
        for key in sorted(self._MODEL_CONTEXT_LIMITS, key=len, reverse=True):
            if key in model_lower:
                return self._MODEL_CONTEXT_LIMITS[key]

        # 兜底:从 env 或默认值
        return int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "4096"))

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        timeout: int | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """根据系统提示和用户提示调用大模型。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.think(
            messages,
            temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stop=stop,
        )

    def chat_turn(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
        timeout: int | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stop: list[str] | None = None,
    ) -> LLMTurn:
        """优先走原生 tools；接口不支持时退回纯文本。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = self.client.with_options(timeout=timeout) if timeout else self.client
        use_tools = bool(tools) and not self._tools_unsupported
        if use_tools:
            create_kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
                "tools": tools,
                "tool_choice": "auto",
            }
            if max_tokens:
                create_kwargs["max_tokens"] = max_tokens
            try:
                print(f"[LLM] 原生 function calling | {self.model}", flush=True)
                response = client.chat.completions.create(**create_kwargs)
                msg = response.choices[0].message if response.choices else None
                turn = parse_assistant_message(msg)
                if turn.tool_name:
                    print(f"[LLM] tool_call: {turn.tool_name} {str(turn.tool_args)[:160]}", flush=True)
                elif turn.content:
                    preview = turn.content.replace("\n", " ")[:160]
                    print(f"[LLM] 无 tool_call，收到文本: {preview}", flush=True)
                return turn
            except Exception as err:
                if _is_tools_unsupported_error(err):
                    print(f"[LLM] 接口不支持 tools，退回 JSON 协议 | {str(err)[:120]}", flush=True)
                    self._tools_unsupported = True
                else:
                    raise
        text = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            json_mode=True,
            stop=stop,
        )
        return LLMTurn(content=text, used_native_tools=False)

    @staticmethod
    def _is_repetition_loop(text: str) -> bool:
        """检测模型把同一句话复读停不下来。"""
        if len(text) < 160:
            return False
        tail = text[-48:]
        if text.count(tail) >= 4:
            return True
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 6 and len(set(lines[-6:])) <= 2:
            return True
        return False

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        timeout: int | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """调用大语言模型。超时且已写出较长正文时保留已生成内容，避免整段重跑。"""
        max_retries = 3
        last_error = None
        client = self.client.with_options(timeout=timeout) if timeout else self.client
        use_json = json_mode
        use_stop = list(stop) if stop else None

        for attempt in range(1, max_retries + 1):
            collected_content: list[str] = []
            try:
                print(f"[LLM] 正在调用 {self.model} 模型 (第 {attempt}/{max_retries} 次)...", flush=True)
                create_kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                }
                if max_tokens:
                    create_kwargs["max_tokens"] = max_tokens
                if use_json:
                    create_kwargs["response_format"] = {"type": "json_object"}
                if use_stop:
                    create_kwargs["stop"] = use_stop
                try:
                    response = client.chat.completions.create(**create_kwargs)
                except Exception as fmt_err:
                    err_l = str(fmt_err).lower()
                    dropped_optional = False
                    if use_json and (
                        "response_format" in err_l
                        or "json_object" in err_l
                        or "unknown parameter" in err_l
                    ):
                        print("[LLM] 接口不支持 json_object，改用普通生成", flush=True)
                        use_json = False
                        create_kwargs.pop("response_format", None)
                        dropped_optional = True
                    if use_stop and ("stop" in err_l or "unknown parameter" in err_l):
                        print("[LLM] 接口不支持 stop，去掉停止词后重试", flush=True)
                        use_stop = None
                        create_kwargs.pop("stop", None)
                        dropped_optional = True
                    if dropped_optional:
                        response = client.chat.completions.create(**create_kwargs)
                    else:
                        raise

                print("[OK] 大语言模型响应成功:", flush=True)
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                    joined = "".join(collected_content)
                    if self._is_repetition_loop(joined):
                        print("\n[LLM] 检测到复读循环，已截断生成本次输出", flush=True)
                        return joined
                print()
                return "".join(collected_content)

            except Exception as e:
                last_error = e
                partial = "".join(collected_content).strip()
                if len(partial) >= 400:
                    print(
                        f"\n[LLM] 生成中断，保留已写出的 {len(partial)} 字（不再整段重跑）",
                        flush=True,
                    )
                    return partial
                err_msg = str(e)[:120]
                if attempt < max_retries:
                    wait = 2 ** attempt
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
