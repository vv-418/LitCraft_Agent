"""
作用：Streamlit 前端主页面，为用户提供可视化的文献综述生成界面。

功能：
1. 输入研究主题、年份过滤、PDF 保存选项
2. 提交后实时轮询显示 Agent 每步执行轨迹（同一页面下方显示）
3. 进度条展示当前执行进度
4. 生成完成后展示 Markdown 综述正文 + 下载 PDF 链接
5. 历史任务记录查看
"""

import os
import time

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# 全局 HTTP Session（绕过代理，避免 CopilotHub 干扰本地通信）
# ---------------------------------------------------------------------------
_session = None


def _api():
    """获取一个绕过代理的 requests.Session 实例。"""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.trust_env = False  # 忽略 HTTP_PROXY / HTTPS_PROXY 环境变量
    return _session

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LitCraft 智能文献综述 Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
API_BASE = os.getenv("LITCRAFT_API_BASE", "http://localhost:8000")
MAX_STEPS = 20  # 与后端 LangGraphAgent max_steps 保持一致


def _check_api() -> bool:
    """检查后端 API 是否可用。"""
    try:
        resp = _api().get(f"{API_BASE}/", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# 侧边栏导航
# ---------------------------------------------------------------------------
st.sidebar.title("📚 LitCraft Agent")
st.sidebar.markdown("智能文献综述生成助手")

page = st.sidebar.radio(
    "导航",
    ["📝 新建综述", "📚 历史记录"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**后端状态：** "
    + ("🟢 运行中" if _check_api() else "🔴 未连接")
)


# ============================================================================
# 页面：新建综述（表单 + 下方实时执行状态）
# ============================================================================
def page_new_task():
    """新建综述 + 执行状态合并页：提交任务后下方直接显示进度。"""
    st.title("📝 新建文献综述")
    st.markdown("输入研究主题，LitCraft Agent 将自动搜索论文、下载 PDF、解析全文、生成综述。")

    # ---------- 表单 ----------
    with st.form("new_task_form"):
        topic = st.text_input(
            "研究主题",
            placeholder="例：Transformer 在自然语言处理中的应用",
            help="输入你想要综述的学术主题，支持中英文",
        )

        year = st.number_input(
            "起始年份（选填，0 表示不限）",
            min_value=0,
            max_value=2026,
            value=0,
            help="输入年份（如 2020）只搜索该年份之后文献；输入 0 则不限年份",
        )

        save_pdf = st.checkbox("📄 生成 PDF 文件", value=True)

        submitted = st.form_submit_button("🚀 开始生成综述", use_container_width=True)

    # ---------- 提交处理 ----------
    if submitted:
        if not topic.strip():
            st.error("请输入研究主题")
            st.stop()

        year_from = str(year) if year > 0 else ""

        with st.spinner("正在提交任务..."):
            try:
                resp = _api().post(
                    f"{API_BASE}/api/agent/start",
                    json={
                        "topic": topic.strip(),
                        "year_from": year_from,
                        "save_pdf": save_pdf,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.task_id = data["task_id"]
                    st.session_state.topic = topic.strip()
                    st.session_state.auto_refresh = True
                    st.success(f"✅ 任务已提交！ID: `{data['task_id'][:8]}...`")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"提交失败：HTTP {resp.status_code}")
            except (requests.ConnectionError, requests.ReadTimeout):
                st.error("❌ 无法连接后端服务，请确认 API 服务已启动（端口 8000）")

    # ---------- 任务执行状态（表单下方） ----------
    task_id = st.session_state.get("task_id")
    if not task_id:
        return

    st.markdown("---")

    try:
        # 查询状态
        status_resp = _api().get(f"{API_BASE}/api/agent/status/{task_id}", timeout=5)
        if status_resp.status_code != 200:
            st.error(f"任务 {task_id} 不存在或被清除")
            st.session_state.auto_refresh = False
            return

        status_data = status_resp.json()
        current_status = status_data["status"]

        # 任务信息头
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**任务 ID：** `{task_id[:8]}...`")
        with col2:
            st.markdown(f"**主题：** {st.session_state.get('topic', '')}")
        with col3:
            if current_status == "running":
                st.info("⏳ 执行中")
            elif current_status == "done":
                st.success("✅ 完成")
            elif current_status == "error":
                st.error("❌ 出错")

                # 显示错误详情（含 traceback）
                err_resp = _api().get(f"{API_BASE}/api/agent/result/{task_id}", timeout=10)
                if err_resp.status_code == 200:
                    err_data = err_resp.json()
                    err_msg = err_data.get("error", "未知错误")
                    tb = err_data.get("traceback")
                    st.markdown(f"**错误信息：** `{err_msg}`")
                    if tb:
                        with st.expander("🔍 查看完整错误堆栈", expanded=False):
                            st.code(tb, language="text")

        # 获取步骤数据
        steps_resp = _api().get(f"{API_BASE}/api/agent/steps/{task_id}", timeout=5)
        steps_data = steps_resp.json().get("steps", []) if steps_resp.status_code == 200 else []

        # ---------- 进度条 ----------
        progress = min(len(steps_data) / MAX_STEPS, 1.0)
        st.progress(progress, text=f"执行进度：{len(steps_data)} / {MAX_STEPS} 步")

        # ---------- 执行轨迹 ----------
        st.subheader("🔄 执行轨迹")

        if not steps_data:
            st.caption("Agent 正在初始化……等待第一步输出")

        for i, step in enumerate(steps_data, start=1):
            with st.expander(f"**步骤 {i}**", expanded=(i == len(steps_data))):
                thought = step.get("thought", "")
                if thought:
                    st.markdown(f"💭 **思考：** {thought}")

                action = step.get("action")
                if action:
                    st.markdown(f"🔧 **行动：** `{action}`")

                action_input = step.get("action_input")
                if action_input:
                    st.code(str(action_input), language="json")

                observation = step.get("observation")
                if observation:
                    st.markdown(f"📥 **观察：**")
                    st.text(observation if len(observation) < 1000 else observation[:1000] + "...")

        # ---------- 完成：展示综述正文 ----------
        if current_status == "done":
            st.markdown("---")
            st.subheader("📄 文献综述正文")

            result_resp = _api().get(f"{API_BASE}/api/agent/result/{task_id}", timeout=10)
            if result_resp.status_code == 200:
                result_data = result_resp.json()
                final_answer = result_data.get("final_answer", "")

                if final_answer:
                    st.markdown(final_answer)
                else:
                    st.warning("最终答案为空")

                pdf_path = result_data.get("pdf_path")
                if pdf_path:
                    filename = os.path.basename(pdf_path)
                    pdf_url = f"{API_BASE}/api/output/{filename}"
                    st.markdown("---")
                    st.markdown(
                        f"<a href='{pdf_url}' target='_blank' style='text-decoration:none;'>"
                        f"<button style='padding:10px 28px;font-size:18px;cursor:pointer;"
                        f"background:#4CAF50;color:white;border:none;border-radius:6px;'>"
                        f"📄 下载 PDF 文件</button></a>",
                        unsafe_allow_html=True,
                    )

            # 允许开始新任务（无论成功或出错）
            if st.button("🔄 生成新的综述", use_container_width=True):
                for key in ["task_id", "topic", "auto_refresh"]:
                    st.session_state.pop(key, None)
                st.rerun()

        # ---------- 自动刷新（运行中时轮询） ----------
        if current_status == "running" and st.session_state.get("auto_refresh", False):
            st.caption("🔄 页面将在 2 秒后自动刷新获取最新进度……")
            time.sleep(2)
            st.rerun()

        if current_status == "running":
            st.button("🔄 手动刷新", on_click=lambda: None)

    except (requests.ConnectionError, requests.ReadTimeout):
        st.error("❌ 无法连接后端服务，请确认 API 服务已启动")
        st.session_state.auto_refresh = False


# ============================================================================
# 页面：历史记录
# ============================================================================
def page_history():
    """历史记录页面：查看以往所有任务的摘要，点击可查看结果。"""
    st.title("📚 历史记录")

    try:
        resp = _api().get(f"{API_BASE}/api/history", timeout=5)
        if resp.status_code != 200:
            st.error("获取历史记录失败")
            return

        tasks = resp.json()
        if not tasks:
            st.info("暂无历史任务记录")
            return

        for task in tasks:
            task_id = task["task_id"]
            topic = task["topic"]
            status = task["status"]
            created_at = task["created_at"]

            tag_map = {"done": "✅ 完成", "error": "❌ 错误"}
            tag = tag_map.get(status, "⏳ 运行中")

            with st.expander(f"[{tag}] {topic}  —  {created_at}"):
                st.markdown(f"**任务 ID：** `{task_id}`")
                st.markdown(f"**主题：** {topic}")
                st.markdown(f"**状态：** {tag}")
                st.markdown(f"**创建时间：** {created_at}")

                if status == "done" and st.button(f"查看结果", key=f"view_{task_id}"):
                    st.session_state.task_id = task_id
                    st.session_state.topic = topic
                    st.session_state.auto_refresh = False
                    st.rerun()

    except (requests.ConnectionError, requests.ReadTimeout):
        st.error("❌ 无法连接后端服务")


# ============================================================================
# 路由
# ============================================================================
if page == "📝 新建综述":
    page_new_task()
elif page == "📚 历史记录":
    page_history()
