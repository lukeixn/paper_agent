from __future__ import annotations

from workflow import run_pipeline


class Kernel:
    def run(self, query: str, top_k: int = 5) -> str:
        return run_pipeline(query, top_k=top_k)


if __name__ == "__main__":
    kernel = Kernel()
    print(kernel.run("请总结视频理解方向最近有哪些值得关注的创新"))
