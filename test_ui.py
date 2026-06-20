from __future__ import annotations

import ui
from streamlit.testing.v1 import AppTest


def test_workflow_hot_reload_guard() -> None:
    module = ui.current_workflow()
    runner = module.run_pipeline_state
    assert "conversation_history" in runner.__annotations__
    assert hasattr(module, "stream_pipeline_state")


def test_offline_analysis_ui() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()

    assert len(app.exception) == 0
    assert [button.label for button in app.button] == ["新建会话"]
    assert len(app.text_area) == 0
    assert [item.placeholder for item in app.chat_input] == [
        "输入研究问题，或继续追问上一轮结果"
    ]
    assert any(
        "实时执行图" in item.value for item in app.markdown
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

    state = app.session_state["analysis_state"]
    assert state["global_context"]["model_config"]["api_key"] == ""
    assert len(app.session_state["chat_messages"]) == 2

    app.chat_input[0].set_value(
        "What methods does it use?"
    ).run(timeout=90)

    assert len(app.exception) == 0
    assert len(app.session_state["chat_messages"]) == 4
    state = app.session_state["analysis_state"]
    assert "Mamba innovation limitation" in state["standalone_query"]
    assert "What methods does it use?" in state["standalone_query"]
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


if __name__ == "__main__":
    test_workflow_hot_reload_guard()
    test_offline_analysis_ui()
    test_library_workspace_is_read_only_management()
    test_search_workspace_contains_online_and_local_import()
    print("streamlit UI tests passed")
