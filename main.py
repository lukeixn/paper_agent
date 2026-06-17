from __future__ import annotations

import argparse
from pathlib import Path

from workflow import run_pipeline as run_workflow_pipeline


def run_pipeline(query: str, top_k: int = 5, data_dir: str = "data") -> str:
    return run_workflow_pipeline(query, top_k=top_k, data_dir=data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper Agent: retrieve papers and run multi-agent analysis.",
    )
    parser.add_argument("query", nargs="*", help="用户问题")
    parser.add_argument("--top-k", type=int, default=5, help="检索论文数量")
    parser.add_argument("--data-dir", default="data", help="论文 JSON 数据目录")
    parser.add_argument("--output", default="", help="可选：报告保存路径")
    parser.add_argument(
        "--require-langgraph",
        action="store_true",
        help="要求必须使用 LangGraph，缺依赖时直接报错",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("请输入你的论文研究问题: ").strip()

    report = run_workflow_pipeline(
        query,
        top_k=args.top_k,
        data_dir=args.data_dir,
        require_langgraph=args.require_langgraph,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf8")

    print(report)


if __name__ == "__main__":
    main()
