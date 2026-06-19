# Paper Agent

一个基于 LangGraph 的论文研究助手：从本地论文库检索材料，路由到多个分析 Agent，并汇总为结构化报告。

## UI 启动

使用项目的 `py311` Conda 环境：

```powershell
D:\py\Anaconda3\envs\py311\python.exe -m streamlit run ui.py
```

浏览器打开 `http://localhost:8501`。API Key 在侧边栏输入，只保存在当前 UI 会话中，不会写入配置文件。

UI 包含两个工作区：

- **研究分析**：检索论文并通过 LangGraph 调度多个 Agent。
- **论文数据库**：查看库中论文，批量上传 PDF，自动解析并重建 FAISS。
- **在线检索**：从 OpenAlex 获取最多 100 篇开放获取候选，经当前
  DeepSeek/OpenAI 模型进行 AI 重排后，单次最多导入 10 篇公开 PDF。

说明：项目不会自动抓取明确禁止爬虫访问的网站，也不会绕过验证码、登录或下载限制。

## 命令行

```powershell
D:\py\Anaconda3\envs\py311\python.exe main.py "Mamba 在长视频理解中的优势和局限" --top-k 10 --require-langgraph
```

## 工作流

```text
START
  -> retrieve
  -> route
  -> survey_agent
  -> innovation_agent
  -> method_agent
  -> limitation_agent
  -> aggregate
  -> END
```

Router 会根据问题选择实际需要执行的 Agent。未选中的 Agent 节点会直接跳过。

## 主要文件

- `ui.py`：Streamlit 用户界面。
- `workflow.py`：LangGraph 工作流。
- `agent/agent.py`：多 Agent 实现。
- `vector_store/search.py`：本地论文检索。
- `aggregator.py`：报告汇总。
- `state/state.py`：共享状态定义。
- `paper_parser.py`：PDF 解析和论文数据生成。

## 安全配置

不要将真实 API Key 写入 `configs/yaml/config.yaml`。可通过 UI 输入，也可设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:OPENAI_API_KEY="..."
```
