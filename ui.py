from __future__ import annotations

import os
from typing import Any

import streamlit as st

from academic_search import AcademicSearchService
from configs.config import cfg
from paper_library import PaperLibrary
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
        st.write("检索 → 路由 → 并行 Agent → ReportAgent")
        if selected_agents:
            st.markdown(
                "\n".join(
                    f"- {AGENT_LABELS.get(name, name)}"
                    for name in selected_agents
                )
            )
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
            width="stretch",
        )


def runtime_model_config(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: settings[key]
        for key in [
            "provider",
            "api_key",
            "model_name",
            "base_url",
            "temperature",
        ]
    }


def has_online_credentials(settings: dict[str, Any]) -> bool:
    if settings["provider"] == "offline":
        return False
    environment_key = (
        "DEEPSEEK_API_KEY"
        if settings["provider"] == "deepseek"
        else "OPENAI_API_KEY"
    )
    return bool(settings["api_key"] or os.getenv(environment_key))


def render_online_academic_search(
    settings: dict[str, Any],
    library: PaperLibrary,
) -> None:
    st.subheader("在线搜索")
    st.caption(
        "从开放学术数据源搜索最多 100 篇候选，经 AI 排名后由你手动选择"
        "最多 10 篇加入论文库。"
    )
    scholar_query = st.text_input(
        "检索内容",
        placeholder="例如：long video understanding memory",
        key="academic_search_query",
    )
    search_clicked = st.button(
        "搜索论文",
        key="academic_search_button",
        type="primary",
        width="stretch",
    )
    if search_clicked:
        if not scholar_query.strip():
            st.warning("请输入论文检索内容。")
        elif not has_online_credentials(settings):
            st.warning("AI 排名需要在线模型和 API Key。")
        else:
            try:
                with st.status(
                    "正在浏览候选论文并进行 AI 排名…",
                    expanded=True,
                ) as status:
                    existing_titles = {
                        paper.title for paper in library.list_papers()
                    }
                    candidates = AcademicSearchService().search(
                        scholar_query,
                        max_candidates=100,
                        existing_titles=existing_titles,
                        model_config=runtime_model_config(settings),
                    )
                    st.session_state["academic_candidates"] = candidates
                    status.update(
                        label=f"排名完成，共获得 {len(candidates)} 篇候选",
                        state="complete",
                        expanded=False,
                    )
            except Exception as exc:
                st.error(f"论文检索失败：{exc}")

    candidates = st.session_state.get("academic_candidates", [])
    if not candidates:
        st.info("输入检索内容后开始搜索。搜索结果不会自动加入论文库。")
        return

    visible_candidates = candidates[:30]
    rows = [
        {
            "选择": False,
            "排名": candidate.rank,
            "标题": candidate.title,
            "年份": candidate.year,
            "作者": "、".join(candidate.authors[:3]),
            "AI 契合度": candidate.final_score,
            "引用": candidate.cited_by_count,
            "公开 PDF": bool(candidate.pdf_url),
            "已在库中": candidate.already_exists,
            "论文页面": candidate.landing_page_url,
        }
        for candidate in visible_candidates
    ]
    edited_rows = st.data_editor(
        rows,
        key="academic_result_editor",
        width="stretch",
        hide_index=True,
        disabled=[
            "排名",
            "标题",
            "年份",
            "作者",
            "AI 契合度",
            "引用",
            "公开 PDF",
            "已在库中",
            "论文页面",
        ],
        column_config={
            "选择": st.column_config.CheckboxColumn(width="small"),
            "排名": st.column_config.NumberColumn(width="small"),
            "标题": st.column_config.TextColumn(width="large"),
            "年份": st.column_config.NumberColumn(width="small"),
            "AI 契合度": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
                width="medium",
            ),
            "公开 PDF": st.column_config.CheckboxColumn(width="small"),
            "已在库中": st.column_config.CheckboxColumn(width="small"),
            "论文页面": st.column_config.LinkColumn(width="small"),
        },
    )
    selected_ranks = {
        int(row["排名"]) for row in edited_rows if row["选择"]
    }
    selected_candidates = [
        candidate
        for candidate in candidates
        if candidate.rank in selected_ranks
    ]
    invalid_candidates = [
        candidate
        for candidate in selected_candidates
        if not candidate.importable
    ]
    st.caption(
        f"已手动选择 {len(selected_candidates)} 篇；"
        "页面展示 AI 排名前 30 篇，单次最多导入 10 篇。"
    )

    import_clicked = st.button(
        "将所选论文加入数据库",
        key="academic_import_button",
        type="primary",
        width="stretch",
        disabled=not selected_candidates,
    )
    if not import_clicked:
        return

    if len(selected_candidates) > 10:
        st.warning("单次最多导入 10 篇，请减少选择。")
    elif invalid_candidates:
        st.warning("选择中包含无公开 PDF 或已存在的论文。")
    elif not has_online_credentials(settings):
        st.warning("导入论文需要在线模型和 API Key。")
    else:
        try:
            with st.status(
                "正在下载、解析并加入论文数据库…",
                expanded=True,
            ) as status:
                results, index_result = library.import_search_candidates(
                    selected_candidates,
                    model_config=runtime_model_config(settings),
                )
                for result in results:
                    prefix = "完成" if result.success else "失败"
                    st.write(f"{prefix}：{result.title or result.filename}")
                status.update(
                    label="所选论文处理完成",
                    state="complete",
                    expanded=True,
                )

            successful = [result for result in results if result.success]
            failed = [result for result in results if not result.success]
            if successful:
                st.success(f"成功加入 {len(successful)} 篇论文。")
            for result in failed:
                st.error(f"{result.filename}：{result.message}")
            if index_result:
                st.caption(
                    "FAISS 已重建："
                    f"{index_result['paper_count']} 篇，"
                    f"{index_result['dimension']} 维。"
                )
        except Exception as exc:
            st.error(f"所选论文导入失败：{exc}")


def render_local_pdf_import(
    settings: dict[str, Any],
    library: PaperLibrary,
) -> None:
    st.subheader("本地 PDF 导入")
    st.caption(
        "适用于需要登录后手动下载的论文。上传后会解析整篇 PDF 的全部页面，"
        "生成结构化信息与向量，并重建 FAISS 索引。"
    )
    uploaded_files = st.file_uploader(
        "选择本地 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key="local_pdf_uploader",
        help="支持一次上传多篇论文。全文会分块解析，不再只读取开头部分。",
    )
    overwrite = st.checkbox(
        "覆盖同名论文",
        value=False,
        key="local_pdf_overwrite",
    )

    if settings["provider"] == "offline":
        st.info("全文解析需要在线 LLM，请选择 DeepSeek 或 OpenAI。")
    elif not has_online_credentials(settings):
        st.info("请在侧边栏输入 API Key 后再导入。")

    import_clicked = st.button(
        "解析全文并加入论文库",
        key="local_pdf_import_button",
        type="primary",
        disabled=not uploaded_files,
        width="stretch",
    )
    if not import_clicked:
        return
    if not has_online_credentials(settings):
        st.warning("导入论文需要可用的在线模型和 API Key。")
        return

    try:
        with st.status("正在分块解析 PDF 全文…", expanded=True) as status:
            st.write(f"准备解析 {len(uploaded_files)} 篇论文")
            results, index_result = library.import_many(
                uploaded_files,
                model_config=runtime_model_config(settings),
                overwrite=overwrite,
            )
            for result in results:
                prefix = "完成" if result.success else "失败"
                st.write(f"{prefix}：{result.title or result.filename}")
            status.update(
                label="本地论文处理完成",
                state="complete",
                expanded=True,
            )

        successful = [result for result in results if result.success]
        failed = [result for result in results if not result.success]
        if successful:
            st.success(f"成功加入 {len(successful)} 篇论文。")
        for result in failed:
            st.error(f"{result.filename}：{result.message}")
        if index_result:
            st.caption(
                "FAISS 已重建："
                f"{index_result['paper_count']} 篇，"
                f"{index_result['dimension']} 维。"
            )
    except Exception as exc:
        st.error(f"本地论文导入失败：{exc}")


def render_academic_search(settings: dict[str, Any]) -> None:
    library = PaperLibrary()
    st.header("论文检索")
    st.caption("在线寻找论文，或将已经下载到本地的 PDF 手动加入论文库。")
    online_tab, local_tab = st.tabs(["在线搜索", "本地 PDF 导入"])
    with online_tab:
        render_online_academic_search(settings, library)
    with local_tab:
        render_local_pdf_import(settings, library)


def render_library(settings: dict[str, Any]) -> None:
    del settings
    library = PaperLibrary()
    stats = library.stats()

    st.header("论文数据库")
    st.caption("查看当前论文库、论文来源和 FAISS 索引状态。")

    columns = st.columns(3)
    columns[0].metric("论文数量", stats["paper_count"])
    columns[1].metric(
        "Embedding 维度",
        stats["embedding_dimension"] or "不一致",
    )
    columns[2].metric(
        "FAISS 索引",
        "可用" if stats["index_exists"] and stats["mapping_exists"] else "未建立",
    )

    papers = library.list_papers()
    search_text = st.text_input(
        "筛选论文",
        placeholder="输入标题、作者或关键词",
    ).strip().lower()
    if search_text:
        papers = [
            paper
            for paper in papers
            if search_text
            in " ".join(
                [
                    paper.title,
                    " ".join(paper.authors),
                    " ".join(paper.keywords),
                ]
            ).lower()
        ]

    rows = [
        {
            "标题": paper.title,
            "作者": "、".join(paper.authors[:4]),
            "关键词": "、".join(paper.keywords[:6]),
            "向量维度": len(paper.embedding),
            "来源": paper.discovery_source or "本地导入",
            "原始页面": paper.source_url,
        }
        for paper in papers
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "标题": st.column_config.TextColumn(width="large"),
            "作者": st.column_config.TextColumn(width="medium"),
            "关键词": st.column_config.TextColumn(width="medium"),
            "向量维度": st.column_config.NumberColumn(width="small"),
            "来源": st.column_config.TextColumn(width="small"),
            "原始页面": st.column_config.LinkColumn(width="small"),
        },
    )

    with st.expander("论文详情"):
        if papers:
            selected_title = st.selectbox(
                "选择论文",
                [paper.title for paper in papers],
            )
            selected = next(
                paper for paper in papers if paper.title == selected_title
            )
            st.subheader(selected.title)
            if selected.authors:
                st.caption("作者：" + "、".join(selected.authors))
            st.write(selected.summary or selected.abstract or "暂无摘要。")
            if selected.contributions:
                st.markdown("**主要贡献**")
                for contribution in selected.contributions:
                    st.markdown(f"- {contribution}")
        else:
            st.info("当前筛选条件下没有论文。")


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
    workspace = st.sidebar.radio(
        "工作区",
        ["研究分析", "论文检索", "论文数据库"],
    )
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

    if workspace == "论文数据库":
        render_library(settings)
        return
    if workspace == "论文检索":
        render_academic_search(settings)
        return

    query = st.text_area(
        "研究问题",
        placeholder="例如：Mamba 在长视频理解中的优势、局限和研究机会是什么？",
        height=120,
    )

    run_clicked = st.button(
        "开始分析",
        type="primary",
        width="stretch",
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
                    model_config=runtime_model_config(settings),
                    require_langgraph=True,
                )
                status.update(label="分析完成", state="complete", expanded=False)
            st.session_state["analysis_state"] = state

    if "analysis_state" in st.session_state:
        st.divider()
        render_result(st.session_state["analysis_state"])


if __name__ == "__main__":
    main()
