# LangGraph Practice Lab

这个目录是一个独立的 LangGraph 练习区，用最小 demo 内化当前项目的多 Agent 工作流。

它不依赖真实论文库、不调用真实 LLM，也不修改主项目业务代码。

## 你会练到什么

| 练习点 | 对应文件 | 对应主项目知识 |
| --- | --- | --- |
| `TypedDict` State | `demo_core.py` | `state/state.py` |
| 节点返回局部更新 | `normalize_question_node`、`retrieve_papers_node` | `workflow.py` 的普通 node |
| Router 选择 Agent | `route_node` | `router.py` |
| `Send` 并行派发 | `dispatch_agents` | `workflow.py` 的 `dispatch_agents` |
| 并行结果合并 | `agent_outputs: Annotated[..., operator.add]` | 主项目的 `agent_outputs` |
| Report 汇总 | `report_node` | `aggregator.py` |
| 实时进度可视化 | `stream_demo`、`app.py` | 主 UI 的实时执行图 |

## 命令行运行

在项目根目录运行：

```powershell
D:\py\Anaconda3\envs\py311\python.exe langgraph_practice\demo_core.py
```

自定义问题：

```powershell
D:\py\Anaconda3\envs\py311\python.exe langgraph_practice\demo_core.py "Which direction is publishable for KV cache compression?"
```

运行后会输出：

1. 节点事件流
2. 最终汇总结果
3. 一个 HTML 可视化文件：`outputs/langgraph_practice_trace.html`

## 可视化页面运行

```powershell
D:\py\Anaconda3\envs\py311\python.exe -m streamlit run langgraph_practice\app.py --server.port 8510
```

然后打开：

```text
http://localhost:8510
```

## 建议练习顺序

1. 先读 `MainState`，理解哪些字段是全局共享的。
2. 读 `normalize_question_node`，理解 node 如何返回局部 state 更新。
3. 读 `route_node`，观察不同问题如何选择不同 Agent。
4. 读 `dispatch_agents`，这是并行的核心。
5. 读 `run_agent_node`，理解每个 Agent 只收到自己的 `AgentTaskState`。
6. 读 `report_node`，理解 parallel outputs 如何被 reduce。
7. 运行 `app.py`，看每个节点如何被高亮。

## 推荐改造练习

### 练习 1：新增一个 Agent

在 `AGENTS` 中加入：

```python
"application_agent": AgentSpec(
    title="Application Agent",
    focus="analyze practical scenarios and product value",
)
```

然后在 `route_node` 中加关键词：

```python
if "application" in query or "应用" in query:
    selected.append("application_agent")
```

目标：理解 Router 和动态可视化如何自动跟随 Agent 数量变化。

### 练习 2：增加一个 State 字段

给 `MainState` 增加：

```python
debug_notes: Annotated[list[str], operator.add]
```

让每个 node 都返回一条 debug note。

目标：理解 LangGraph 的 state 合并规则。

### 练习 3：观察并行输出顺序

在 `run_agent_node` 里给不同 agent 设置不同 `sleep` 时间。

目标：理解并行完成顺序和最终排序不是同一件事。

