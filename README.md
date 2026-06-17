# Paper Agent

一个面向论文库的多 Agent 原型：根据用户问题检索相关论文，然后路由到多个分析 Agent，最后汇总成报告。

## 当前能力

- 从 `data/*.json` 加载已有论文结构化数据。
- 本地离线检索，不依赖 FAISS 或 embedding 模型也能运行主流程。
- Router 根据问题意图选择综述、创新点、方法路线、局限机会等 Agent。
- 多 Agent 输出统一汇总为 Markdown 报告。
- 配置在线 LLM 后，Agent 会优先使用大模型生成更强的分析。

## 快速运行

```bash
python main.py "视频理解方向最近有哪些值得关注的创新？" --top-k 5
```

保存报告：

```bash
python main.py "Mamba 在长视频理解里的优势和局限" --top-k 5 --output reports/mamba_video.md
```

## 在线 LLM

默认使用 DeepSeek 模式。要使用 DeepSeek 或 OpenAI：

1. 在 `configs/yaml/config.yaml` 里确认 `MODEL.PROVIDER` 是 `deepseek` 或 `openai`。
2. 设置环境变量：

```bash
set DEEPSEEK_API_KEY=你的key
set OPENAI_API_KEY=你的key
```

不要把真实 key 写进配置文件。

## 主要目录

- `main.py`: 主流程入口。
- `vector_store/search.py`: 本地论文检索。
- `router.py`: 问题路由。
- `agent/agent.py`: 多 Agent 实现。
- `aggregator.py`: 报告汇总。
- `state/state.py`: 主状态结构。
- `paper_parser.py`: PDF 到 JSON 的解析工具。
