from __future__ import annotations

import base64
import re

import ui
from streamlit.testing.v1 import AppTest


def test_workflow_hot_reload_guard() -> None:
    module = ui.current_workflow()
    runner = module.run_pipeline_state
    assert "conversation_history" in runner.__annotations__
    assert hasattr(module, "stream_pipeline_state")


def test_topology_layout_matches_selected_agent_count() -> None:
    assert ui.topology_agent_layout([]) == []
    assert ui.topology_agent_layout(["survey_agent"]) == [
        ("survey_agent", 210, 390)
    ]
    assert [x for _, x, _ in ui.topology_agent_layout(
        ["survey_agent", "method_agent", "limitation_agent"]
    )] == [100, 210, 320]


def test_conversation_title_is_compact() -> None:
    assert ui.conversation_title("  KV 缓存   优化方向  ") == "KV 缓存 优化方向"
    assert ui.conversation_title("a" * 30) == "a" * 23 + "…"


def topology_svg(app: AppTest) -> str:
    markup = next(
        item.value
        for item in app.markdown
        if "data:image/svg+xml;base64" in item.value
    )
    encoded = re.search(
        r"data:image/svg\+xml;base64,([^\"']+)",
        markup,
    ).group(1)
    return base64.b64decode(encoded).decode("utf-8")


def test_offline_analysis_ui() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()

    assert len(app.exception) == 0
    assert [button.label for button in app.button] == [
        "新建会话",
        "新会话",
    ]
    assert len(app.text_area) == 0
    assert [item.placeholder for item in app.chat_input] == [
        "在当前会话中继续提问"
    ]
    assert any(
        "实时执行图" in item.value for item in app.markdown
    )
    assert any(
        'stColumn"]:has(.workflow-panel)' in item.value
        and "position: sticky" in item.value
        for item in app.markdown
    )

    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("研究分析").run()
    assert app.slider[1].max == 30
    app.slider[1].set_value(10).run()
    app.chat_input[0].set_value(
        "Mamba innovation limitation"
    ).run(timeout=90)

    assert len(app.exception) == 0
    assert any(
        "topology-image" in item.value
        and "data:image/svg+xml;base64" in item.value
        and "<svg" not in item.value
        for item in app.markdown
    )
    assert [tab.label for tab in app.tabs] == [
        "概览",
        "论文",
        "Agent 分析",
        "完整报告",
    ]

    metrics = {metric.label: metric.value for metric in app.metric}
    assert 1 <= int(metrics["检索论文"]) <= 10
    assert metrics["执行 Agent"] == "2"
    assert metrics["路由"] == "multi_agent"
    svg = topology_svg(app)
    assert "创新分析" in svg
    assert "局限与机会" in svg
    assert "研究综述" not in svg
    assert "方法比较" not in svg
    assert svg.count('<g class="topo-agent') == 2

    sessions = app.session_state["conversation_sessions"]
    active_id = app.session_state["active_conversation_id"]
    state = sessions[active_id]["analysis_state"]
    assert state["global_context"]["model_config"]["api_key"] == ""
    assert len(sessions[active_id]["messages"]) == 2
    assert sessions[active_id]["title"] == "Mamba innovation limita…"

    app.chat_input[0].set_value(
        "What methods does it use?"
    ).run(timeout=90)

    assert len(app.exception) == 0
    sessions = app.session_state["conversation_sessions"]
    active_id = app.session_state["active_conversation_id"]
    assert len(sessions[active_id]["messages"]) == 4
    state = sessions[active_id]["analysis_state"]
    assert "Mamba innovation limitation" in state["standalone_query"]
    assert "What methods does it use?" in state["standalone_query"]
    assert len(app.chat_message) == 4

    old_conversation_id = active_id
    next(
        button for button in app.button
        if button.label == "新建会话"
    ).click().run()

    new_conversation_id = app.session_state["active_conversation_id"]
    assert new_conversation_id != old_conversation_id
    assert len(app.chat_message) == 0
    assert len(app.session_state["conversation_sessions"]) == 2

    app.chat_input[0].set_value(
        "A completely separate topic"
    ).run(timeout=90)
    sessions = app.session_state["conversation_sessions"]
    assert len(sessions[new_conversation_id]["messages"]) == 2
    assert len(sessions[old_conversation_id]["messages"]) == 4

    next(
        button for button in app.button
        if button.label == "Mamba innovation limita…"
    ).click().run()
    assert app.session_state[
        "active_conversation_id"
    ] == old_conversation_id
    assert len(app.chat_message) == 4


def test_library_workspace_is_read_only_management() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()
    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("论文数据库").run()

    assert len(app.exception) == 0
    assert len(app.file_uploader) == 0
    assert len(app.tabs) == 0
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["论文数量"]) > 0
    assert metrics["FAISS 索引"] == "可用"


def test_search_workspace_contains_online_and_local_import() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()
    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("论文检索").run()

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "在线搜索",
        "本地 PDF 导入",
    ]
    assert "检索内容" in [text.label for text in app.text_input]
    assert "搜索论文" in [button.label for button in app.button]
    assert [uploader.label for uploader in app.file_uploader] == [
        "选择本地 PDF"
    ]
    assert "解析全文并加入论文库" in [
        button.label for button in app.button
    ]


def test_agent_skills_workspace() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()
    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("Agent Skills").run()

    assert len(app.exception) == 0
    assert "目标 Agent" in [item.label for item in app.selectbox]
    assert [uploader.label for uploader in app.file_uploader] == [
        "选择 Markdown Skill"
    ]
    assert "安装 Skill" in [button.label for button in app.button]
    assert len(app.dataframe) == 1
    skill_rows = app.dataframe[0].value
    assert "innovation.md" in skill_rows["文件"].tolist()
    assert "内置" in skill_rows["来源"].tolist()


if __name__ == "__main__":
    test_workflow_hot_reload_guard()
    test_topology_layout_matches_selected_agent_count()
    test_conversation_title_is_compact()
    test_offline_analysis_ui()
    test_library_workspace_is_read_only_management()
    test_search_workspace_contains_online_and_local_import()
    test_agent_skills_workspace()
    print("streamlit UI tests passed")
