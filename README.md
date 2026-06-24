# Paper Agent

Paper Agent 是一个面向论文阅读、论文库管理和研究方向分析的本地 AI Agent 工具。

它提供一个 Streamlit 图形界面，可以完成：

- 本地论文库检索与多 Agent 分析
- ChatGPT 风格的连续研究会话
- 本地 PDF 全文解析并加入论文库
- 在线论文搜索后手动选择加入论文库
- 查看、删除论文库中的论文
- 自动重建 FAISS 向量索引
- 为不同 Agent 导入和编辑 Markdown Skill

## 1. 环境要求

推荐使用项目开发时的环境：

```text
Python 3.11
conda 环境名：py311
```

主要依赖：

```text
streamlit
langgraph
langchain-core
openai
pymupdf
sentence-transformers
faiss-cpu
pydantic
pyyaml
numpy
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如果你使用的是本机已有的 conda 环境，请先切换到对应环境：

```powershell
conda activate py311
```

## 2. 启动项目

### 方式一：双击启动

在项目根目录双击：

```text
start_ui.bat
```

然后访问：

```text
http://localhost:8502
```

### 方式二：命令行启动

```powershell
D:\py\Anaconda3\envs\py311\python.exe -m streamlit run ui.py --server.port 8502
```

如果你的 Python 路径不同，请把命令里的 Python 路径替换成自己的环境路径。

## 3. 配置模型

打开 UI 后，在左侧栏配置：

1. 选择服务商：`DeepSeek`、`OpenAI` 或 `Offline`
2. 输入 API Key
3. 设置模型名、API 地址、温度
4. 设置每次检索论文数量

API Key 只保存在当前浏览器会话中，不会写入配置文件。

也可以通过环境变量配置：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
$env:OPENAI_API_KEY="你的 key"
```

不要把真实 API Key 写入 `configs/yaml/config.yaml` 后提交到 GitHub。

## 4. 工作区说明

左侧“工作区”包含四个主要功能：

```text
研究分析
论文检索
论文数据库
Agent Skills
```

### 4.1 研究分析

用于向当前论文库提问。

基本流程：

1. 输入研究问题
2. 系统从本地论文库检索相关论文
3. Router 判断需要调用哪些专家 Agent
4. 多个 Agent 并行分析
5. ReportAgent 汇总最终回答

当前支持连续会话：

- 同一个会话中可以直接追问
- 历史用户问题会进入 LangGraph state
- 新研究主题建议点击“新建会话”

右侧实时执行图会显示当前执行到哪个节点，以及调用了哪些 Agent。

### 4.2 论文检索

论文检索分为两个标签页：

```text
在线搜索
本地 PDF 导入
```

#### 在线搜索

用于从开放学术数据源搜索论文。

使用方式：

1. 输入搜索关键词
2. 点击“搜索论文”
3. 系统浏览最多 100 篇候选论文
4. AI 根据相关性排序
5. 你手动勾选要加入论文库的论文
6. 单次最多导入 10 篇

注意：

- 在线搜索不会在提问时自动触发
- 必须由用户手动搜索、手动选择、手动导入
- 没有公开 PDF 的论文无法自动导入

#### 本地 PDF 导入

用于导入你已经手动下载到本地的 PDF。

适合这些情况：

- 论文网站需要登录
- 论文来自学校数据库
- 在线搜索无法直接下载 PDF
- 你已经有本地 PDF 文件

使用方式：

1. 进入“论文检索”
2. 选择“本地 PDF 导入”
3. 上传一个或多个 PDF
4. 根据需要勾选“覆盖同名论文”
5. 点击“解析全文并加入论文库”

导入后系统会：

1. 读取 PDF 的全部可提取文本页
2. 按页顺序切分成文本块
3. 调用 LLM 提取标题、摘要、贡献、局限、关键词等结构化信息
4. 生成 embedding
5. 保存论文 JSON
6. 保存 PDF 文件
7. 重建 FAISS 索引

如果 PDF 是扫描版，没有文本层，当前版本会提示无法提取文本。后续可以扩展 OCR。

### 4.3 论文数据库

用于查看和管理当前论文库。

可以查看：

- 论文数量
- embedding 维度
- FAISS 索引状态
- 论文标题
- 作者
- 关键词
- 来源
- 原始页面

可以删除论文：

1. 在论文表格中勾选“删除”
2. 勾选“我确认删除所选论文”
3. 点击“删除所选论文”

删除时会同步处理：

- 删除论文 JSON
- 删除对应 PDF
- 重建 `data/faiss.index`
- 重建 `data/id_mapping.json`

如果论文库被删空，旧的 FAISS 索引文件也会被清理，避免残留向量影响后续检索。

### 4.4 Agent Skills

用于管理不同 Agent 的 Markdown Skill。

项目内置四类专家 Agent：

```text
survey_agent       研究综述
innovation_agent   创新分析
method_agent       方法比较
limitation_agent   局限与机会
```

每个 Agent 只读取自己的 Skill。

使用方式：

1. 进入“Agent Skills”
2. 选择目标 Agent
3. 上传 `.md` 或 `.markdown` 文件
4. 点击“安装 Skill”
5. 下一次运行该 Agent 时自动生效

外部 Skill 保存位置：

```text
agent_skills/
  survey_agent/
  innovation_agent/
  method_agent/
  limitation_agent/
```

内置 Skill 保存位置：

```text
profiles/
  survey.md
  innovation.md
  method.md
  limitation.md
```

UI 中可以打开并编辑 Skill 内容。保存后，下次 Agent 执行时生效。

如果外部 Skill 与内置 Skill 同名，外部 Skill 会覆盖内置 Skill。删除外部同名文件后，内置 Skill 会恢复生效。

## 5. 项目运行架构

核心流程基于 LangGraph：

```text
START
  -> contextualize
  -> retrieve
  -> route
  -> parallel agents
       survey_agent
       innovation_agent
       method_agent
       limitation_agent
  -> report_agent
  -> END
```

各节点职责：

- `contextualize`：结合当前会话历史，把追问改写成可独立检索的问题
- `retrieve`：从本地论文库检索相关论文
- `route`：判断本轮需要哪些 Agent
- `run_agent`：并行执行专家 Agent
- `report_agent`：汇总 Agent 输出，生成最终回答

多轮会话中：

- `query` 表示当前用户问题
- `user_question_history` 保存历史用户问题
- 不会把完整历史报告塞进每个 Agent 上下文
- 每个 Agent 只看自己的任务上下文
- 最终回答由 ReportAgent 汇总

## 6. 数据目录说明

常用目录：

```text
data/               论文结构化 JSON、FAISS 索引、id_mapping
papers/             保存导入的 PDF
profiles/           内置 Agent Skill
agent_skills/       用户上传的外部 Agent Skill
reports/            可选报告输出
langgraph_practice/ LangGraph 练习区
```

主要索引文件：

```text
data/faiss.index
data/id_mapping.json
```

论文库发生导入或删除后，会自动重建这两个文件。

## 7. 常见问题

### 7.1 页面打不开

确认 Streamlit 是否正在运行。

如果使用 `start_ui.bat`，命令行窗口需要保持打开。

默认地址：

```text
http://localhost:8502
```

### 7.2 API Key 输入后还不能用

检查：

- 服务商是否选对
- API 地址是否正确
- 模型名是否正确
- Key 是否有余额或权限

DeepSeek 默认地址：

```text
https://api.deepseek.com
```

### 7.3 PDF 导入失败

常见原因：

- PDF 是扫描版，没有文本层
- API Key 无效或网络失败
- 模型返回的结构化 JSON 不合法
- 本地 embedding 模型不可用

当前版本已经增加 JSON 修复机制，模型输出 Markdown 代码块、前后说明、部分格式错误时，会尝试自动修复一次。

### 7.4 删除论文后重新导入还提示已存在

新版已修复 PDF 残留问题。

删除论文时会记录并使用 `pdf_filename` 删除对应 PDF；如果历史版本留下了孤儿 PDF，重新导入时会允许覆盖。

### 7.5 想重新生成索引

导入或删除论文会自动重建 FAISS。

如果需要手动重建，可以在代码中调用：

```python
from paper_library import PaperLibrary

PaperLibrary().rebuild_index()
```

## 8. LangGraph 练习区

项目包含一个独立练习目录：

```text
langgraph_practice/
```

运行练习 UI：

```powershell
D:\py\Anaconda3\envs\py311\python.exe -m streamlit run langgraph_practice\app.py --server.port 8510
```

命令行运行 demo：

```powershell
D:\py\Anaconda3\envs\py311\python.exe langgraph_practice\demo_core.py
```

这个练习区用于理解：

- `StateGraph`
- LangGraph State
- Node
- Conditional Edge
- `Send` 并行
- `operator.add` 合并并行结果
- `stream` 实时执行事件

## 9. 测试

常用测试命令：

```powershell
D:\py\Anaconda3\envs\py311\python.exe test_workflow.py
D:\py\Anaconda3\envs\py311\python.exe test_paper_library.py
D:\py\Anaconda3\envs\py311\python.exe test_full_text_parser.py
D:\py\Anaconda3\envs\py311\python.exe test_skill_library.py
D:\py\Anaconda3\envs\py311\python.exe test_ui.py
```

## 10. 安全提醒

- 不要把真实 API Key 提交到 GitHub
- 不要公开上传含版权风险的 PDF
- 如果要商业分发，建议把模型 Key、数据库、用户上传文件和应用代码分离管理
- `data/` 和 `papers/` 中的内容可能包含论文全文或派生数据，公开前请确认版权和隐私风险
