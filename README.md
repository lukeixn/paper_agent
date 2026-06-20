# Paper Agent

一个基于 LangGraph 的论文研究助手，支持本地知识库分析、在线论文检索、
本地 PDF 导入和多 Agent 并行研究报告生成。

## UI 启动

双击项目根目录的 `start_ui.bat`，然后访问：

```text
http://localhost:8502
```

也可以直接运行：

```powershell
D:\py\Anaconda3\envs\py311\python.exe -m streamlit run ui.py --server.port 8502
```

## 三个工作区

- **研究分析**：只检索本地论文库，通过 LangGraph 并行执行专业 Agent。
- **论文检索**
  - 在线搜索：从 OpenAlex 获取候选，AI 排序后手动选择导入。
  - 本地 PDF 导入：用于已经从登录网站手动下载的论文。
- **论文数据库**：查看已有论文、来源和 FAISS 索引状态。

## PDF 全文解析

PDF 导入不再只截取开头内容。当前流程为：

1. 读取 PDF 的所有可提取文本页面。
2. 按页面顺序分块，确保全部内容被覆盖。
3. 每个文本块独立提取方法、结果、贡献和局限。
4. 汇总所有分块笔记，生成最终结构化论文信息。
5. 生成 embedding 并重建 FAISS。

扫描版 PDF 如果没有文本层，当前版本会提示无法提取文本；后续可增加 OCR。

## LangGraph 工作流

```text
START
  -> retrieve
  -> route
  -> [并行分支]
       survey_agent
       innovation_agent
       method_agent
       limitation_agent
  -> report_agent
  -> END
```

专业 Agent 仅接收用户问题、检索论文、模型配置和自身任务，无法读取其他
Agent 的上下文或输出。所有分支结束后由 `ReportAgent` 汇总。

## 安全配置

不要将真实 API Key 写入 `configs/yaml/config.yaml`。可通过 UI 输入，或设置：

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:OPENAI_API_KEY="..."
```
