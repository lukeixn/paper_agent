from __future__ import annotations

from workflow import run_pipeline, workflow_info


def test_workflow_info() -> None:
    info = workflow_info()
    assert "retrieve" in info["nodes"]
    assert "route" in info["nodes"]
    assert "aggregate" in info["nodes"]
    assert ("retrieve", "route") in info["edges"]


def test_pipeline_generates_report() -> None:
    report = run_pipeline("Mamba 在长视频理解里的优势和局限", top_k=3)
    assert "# 论文 Agent 分析报告" in report
    assert "VAMBA" in report or "Video Mamba" in report
    assert "innovation_agent" in report
    assert "limitation_agent" in report


if __name__ == "__main__":
    test_workflow_info()
    test_pipeline_generates_report()
    print("workflow tests passed")
