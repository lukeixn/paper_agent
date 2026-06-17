from __future__ import annotations

import argparse
from pathlib import Path

from agent.agent import AGENT_REGISTRY
from aggregator import ReportAggregator
from router import Router
from state.state import create_state
from vector_store.search import PaperSearchEngine


def run_pipeline(query: str, top_k: int = 5, data_dir: str = "data") -> str:
    state = create_state(query)

    search_engine = PaperSearchEngine(data_dir=data_dir)
    state["retrieved_papers"] = search_engine.search(query, top_k=top_k)

    route_decision = Router().route(query)
    state["route"] = route_decision.route
    state["global_context"]["route_reason"] = route_decision.reason

    for agent_name in route_decision.agents:
        agent_cls = AGENT_REGISTRY[agent_name]
        state = agent_cls()(state)

    report = ReportAggregator().build_report(state)
    state["final_report"] = report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper Agent: retrieve papers and run multi-agent analysis.",
    )
    parser.add_argument("query", nargs="*", help="用户问题")
    parser.add_argument("--top-k", type=int, default=5, help="检索论文数量")
    parser.add_argument("--data-dir", default="data", help="论文 JSON 数据目录")
    parser.add_argument("--output", default="", help="可选：报告保存路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("请输入你的论文研究问题: ").strip()

    report = run_pipeline(query, top_k=args.top_k, data_dir=args.data_dir)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf8")

    print(report)


if __name__ == "__main__":
    main()
