from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_offline_analysis_ui() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()

    assert len(app.exception) == 0
    assert [button.label for button in app.button] == ["开始分析"]
    assert [text_area.label for text_area in app.text_area] == ["研究问题"]

    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("研究分析").run()
    app.text_area[0].set_value("Mamba innovation limitation").run()
    app.slider[1].set_value(10).run()
    app.button[0].click().run(timeout=90)

    assert len(app.exception) == 0
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


def test_library_workspace() -> None:
    app = AppTest.from_file("ui.py", default_timeout=90)
    app.run()
    app.selectbox[0].set_value("offline").run()
    app.radio[0].set_value("论文数据库").run()

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == ["导入论文", "当前论文"]
    assert [uploader.label for uploader in app.file_uploader] == ["选择 PDF"]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert int(metrics["论文数量"]) > 0
    assert metrics["FAISS 索引"] == "可用"


if __name__ == "__main__":
    test_offline_analysis_ui()
    test_library_workspace()
    print("streamlit UI tests passed")
