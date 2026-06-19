from __future__ import annotations

import os
from typing import Any

import streamlit as st

from configs.config import cfg
from workflow import langgraph_available, run_pipeline_state


AGENT_LABELS = {
    "survey_agent": "研究综述",
    "innovation_agent": "创新分析",
    "method_agent": "方法比较",
    "limitation_agent": "局限与机会",
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { color: #1e2926; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            border-right: 1px solid #dde3df;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 720;
            line-height: 1.15;
            margin: 0;
        }
        .app-subtitle {
            color: #60706a;
            margin: .4rem 0 1.6rem;
        }
        .status-line {
            display: flex;
            gap: .65rem;
            align-items: center;
            color: #53635d;
            font-size: .9rem;
            margin-bottom: 1rem;
        }
        .status-dot {
            width: .55rem;
            height: .55rem;
            border-radius: 50%;
            background: #147d64;
            display: inline-block;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #dde3df;
            border-radius: 6px;
            padding: .8rem 1rem;
            background: #fff;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #dde3df;
            border-radius: 6px;
            background: #fff;
        }
        .paper-score {
            color: #147d64;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_model(provider: str) -> str:
    provider_key = provider.upper()
    return cfg["MODEL"].get(provider_key, {}).get("MODEL_NAME", "")


def model_settings() -> dict[str, Any]:
    st.sidebar.subheader("模型连接")
    provider = st.sidebar.selectbox(
        "服务商",
        ["deepseek", "openai", "offline"],
        format_func=lambda value: {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "offline": "离线模式",
        }[value],
    )

    api_key = ""
    model_name = ""
    base_url = ""
    if provider != "offline":
        api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            placeholder="仅在当前会话中使用",
        )
        model_name = st.sidebar.text_input(
            "模型",
            value=default_model(provider),
        )
        if provider == "deepseek":
            base_url = st.sidebar.text_input(
                "API 地址",
                value=cfg["MODEL"]["DEEPSEEK"]["BASE_URL"],
            )

    temperature = st.sidebar.slider(
        "温度",
        min_value=0.0,
        max_value=1.5,
        value=float(cfg["MODEL"].get("TEMPERATURE", 0.3)),
        step=0.1,
    )
    top_k = st.sidebar.slider("检索论文数", 1, 20, 10)

    return {
        "provider": provider,
        "api_key": api_key.strip(),
        "model_name": model_name.strip(),
        "base_url": base_url.strip(),
        "temperature": temperature,
        "top_k": top_k,
    }


def render_papers(state: dict[str, Any]) -> None:
    papers = state.get("retrieved_papers", [])
    if not papers:
        st.info("没有检索到相关论文。")
        return

    for index, paper in enumerate(papers, start=1):
        with st.expander(f"{index}. {paper.title}", expanded=index <= 2):
            st.markdown(
                f'<span class="paper-score">相关度 {paper.score:.4f}</span>',
                unsafe_allow_html=True,
            )
            if paper.authors:
                st.caption("作者：" + "、".join(paper.authors))
            st.write(paper.summary or paper.abstract or "暂无摘要。")
            if paper.keywords:
                st.caption("关键词：" + " · ".join(paper.keywords[:8]))


def render_agents(state: dict[str, Any]) -> None:
    results = state.get("agent_results", {})
    if not results:
        st.info("本次问题没有触发分析 Agent。")
        return

    for agent_name, result in results.items():
        st.subheader(AGENT_LABELS.get(agent_name, agent_name))
        st.markdown(result)


def render_result(state: dict[str, Any]) -> None:
    papers = state.get("retrieved_papers", [])
    results = state.get("agent_results", {})
    selected_agents = state.get("selected_agents", [])

    metric_columns = st.columns(3)
    metric_columns[0].metric("检索论文", len(papers))
    metric_columns[1].metric("执行 Agent", len(results))
    metric_columns[2].metric("路由", state.get("route", ""))

    tabs = st.tabs(["概览", "论文", "Agent 分析", "完整报告"])
    with tabs[0]:
        st.subheader("执行路径")
        st.write(" → ".join(["检索", "路由", *[
            AGENT_LABELS.get(name, name) for name in selected_agents
        ], "汇总"]))
        reason = state.get("global_context", {}).get("route_reason")
        if reason:
            st.caption(reason)
        errors = state.get("errors", [])
        if errors:
            for error in errors:
                st.warning(error)

    with tabs[1]:
        render_papers(state)

    with tabs[2]:
        render_agents(state)

    with tabs[3]:
        report = state.get("final_report", "")
        st.markdown(report)
        st.download_button(
            "下载 Markdown 报告",
            data=report.encode("utf-8"),
            file_name="paper-agent-report.md",
            mime="text/markdown",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Paper Agent",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    settings = model_settings()
    st.sidebar.divider()
    st.sidebar.caption(
        "API Key 只保存在当前浏览器会话中，不会写入配置文件或报告。"
    )

    st.markdown('<h1 class="app-title">Paper Agent</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">检索论文库，通过 LangGraph 调度多个研究 Agent，并生成结构化报告。</p>',
        unsafe_allow_html=True,
    )
    graph_status = "LangGraph 已连接" if langgraph_available() else "兼容模式"
    st.markdown(
        f'<div class="status-line"><span class="status-dot"></span>{graph_status}</div>',
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "研究问题",
        placeholder="例如：Mamba 在长视频理解中的优势、局限和研究机会是什么？",
        height=120,
    )

    run_clicked = st.button(
        "开始分析",
        type="primary",
        use_container_width=True,
    )
    if run_clicked:
        if not query.strip():
            st.warning("请先输入研究问题。")
        elif settings["provider"] != "offline" and not (
            settings["api_key"]
            or os.getenv(
                "DEEPSEEK_API_KEY"
                if settings["provider"] == "deepseek"
                else "OPENAI_API_KEY"
            )
        ):
            st.warning("请输入 API Key，或选择离线模式。")
        else:
            with st.status("正在运行论文分析工作流…", expanded=True) as status:
                st.write("检索相关论文")
                st.write("路由分析任务")
                state = run_pipeline_state(
                    query.strip(),
                    top_k=settings["top_k"],
                    model_config={
                        key: settings[key]
                        for key in [
                            "provider",
                            "api_key",
                            "model_name",
                            "base_url",
                            "temperature",
                        ]
                    },
                    require_langgraph=True,
                )
                status.update(label="分析完成", state="complete", expanded=False)
            st.session_state["analysis_state"] = state

    if "analysis_state" in st.session_state:
        st.divider()
        render_result(st.session_state["analysis_state"])


if __name__ == "__main__":
    main()
