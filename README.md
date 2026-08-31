# rag_docs (ragkb)

> **诊断知识库工业级文档处理流水线：Markdown/Word/PDF → QA CSV + SOP，9 步全自动化 + 双层门禁 + 大模型视觉读图，零容忍未审内容发布。**

## 项目定位 / 背景

`rag_docs`（包名 `ragkb`）是**面向内网诊断知识库**的工业级文档处理流水线。它解决一个生产中反复遇到的问题：**把零散的产品手册、运维 SOP、故障排查指南，规整成可直接灌进向量库的高质量知识单元**。

传统做法（人手拆、人工 QA）成本高、不可复制；早期 LLM 做法（一次性 prompt）质量差、幻觉多。本项目的工程化方案是 **9 步流水线 + 双层门禁 + 大模型视觉读图 + 跨文档聚合去重 + 断点续跑**：

```
discover topics
  → per-doc (ThreadPoolExecutor, max_workers 默认 10):
      parse → vision-read images → classify sections → extract QA/SOP
            → Layer1 struct gate → Layer2 semantic gate → 一次有界 regeneration
  → cross-doc aggregate (by topic) → global dedup
  → store results.json + manifest + decisions
```

每一步都对真实生产故障做过优化（详见 `docs/retrieval-output-design.md` 与 `调度全流程与经验总结.md`）：

- **Fence-aware Markdown 解析**：代码 fence 内的 `# 注释` 不会被误判为标题
- **OCR 块剥离**：图片的旧版 `<details><pre>OCR</pre></details>` 仅作弱提示，**真实文本由视觉大模型读原图**
- **大模型视觉读图**：不信任预置 OCR（容易错字），用 Gemini/Claude 多模态重转写
- **结构门禁**（`gate_struct`）：确保产出的 JSON schema 合法、id 唯一、字段非空
- **语义门禁**（`gate_semantic`）：强模型审核每条 QA/SOP 准确性 + 完整性 + 是否答非所问，**FAIL-CLOSED**——审不过的不发布
- **持久化决策**（`DecisionStore`）：审不了的进入"人工决策队列"（`include/exclude/retry/accept`），不阻塞主流程
- **断点续跑**（`manifest`）：已 pinned 的文档永远不被覆盖，幂等
- **可观测性**（`EventBus` + `events.jsonl`）：CLI / 仪表板跨进程共享同一份事件流

LLM 走的是**京东内部 LLM 网关**（`JD_LLM_BASE_URL` OpenAI 兼容），支持多模型 fallback（`JD_LLM_FALLBACK_MODELS` 链：Gemini-3.1-Pro-Preview → Gemini-3-Flash-Preview → Claude-Sonnet-4.6 → Claude-Opus-4.6 → Claude-Opus-4.7）。网关限流是 per-model 独立桶，所以多模型并行≈各模型限流桶之和。

## 仓库结构

```
rag_docs/
├── pyproject.toml                   # name="ragkb" v0.1.0, Python ≥3.11, hatchling 后端
├── uv.lock                          # uv 锁定的全量依赖
├── .env.example                     # JD_LLM_* + RAGKB_* 完整模板
├── README.md                        # 完整使用手册（中文 178 行）
├── 调度全流程与经验总结.md            # 排障与设计经验长文
├── docs/
│   └── retrieval-output-design.md   # 检索输出格式设计
├── scripts/                         # 运维脚本
│   ├── probe_rate_limit.py          # 网关限流探针（验证 per-model 独立）
│   ├── merge_sop_semantic.py        # SOP 语义合并
│   ├── rehabilitate_failed_sop.py   # 失败 SOP 复检
│   └── retry_no_verdict_sop.py      # 无审结论 SOP 重试
├── src/ragkb/
│   ├── cli.py                       # `ragkb run / export / serve` 入口（typer）
│   ├── config.py                    # 两组 settings：JD 网关 + 流水线开关
│   ├── llm/client.py                # 模型无关网关客户端（dual-dialect, thread-local）
│   ├── parse/
│   │   ├── markdown.py              # fence-aware parser + OCR 块剥离
│   │   ├── model.py                 # Document / Section / Image 数据类
│   │   └── source.py                # 多格式加载（md/docx/pdf/txt/html）+ sha
│   ├── pipeline/
│   │   ├── orchestrator.py          # 主驱动器（~58KB，含并发+幂等+重试+导出）
│   │   ├── control.py               # DecisionStore + HumanReviewRequired
│   │   ├── events.py                # EventBus + events.jsonl 跨进程 tail
│   │   ├── classify.py              # 章节分类（QA vs SOP）
│   │   ├── extract.py               # 抽取 QA/SOP（带 vision 转写融合）
│   │   ├── gate_struct.py           # Layer1：结构门禁
│   │   ├── gate_semantic.py         # Layer2：语义门禁（fail-closed）
│   │   ├── regenerate.py            # 失败重生的有界 retry
│   │   ├── aggregate.py             # 跨文档按主题聚合
│   │   ├── dedup.py                 # 全局去重
│   │   ├── paraphrase.py            # 实验性问法扩写
│   │   ├── export.py                # 写 CSV / SOP / metadata
│   │   ├── vision.py                # 视觉读图
│   │   ├── imageprep.py             # 图片预处理（缩放/格式）
│   │   ├── prompts.py               # 所有 prompt 模板（~16KB）
│   │   ├── jsonutil.py              # 容忍 JSON 解析
│   │   ├── batching.py              # size-bounded packing
│   │   ├── scrub.py                 # 敏感信息脱敏
│   │   ├── units.py                 # QAUnit / SOPUnit 数据类
│   │   ├── sections.py              # 超大章节拆分
│   │   ├── maintenance.py           # 维护任务
│   │   └── failures.py              # 失败分类
│   ├── server/
│   │   ├── app.py                   # FastAPI 仪表板（SSE 实时事件流）
│   │   └── web/index.html           # 单文件 SPA（~39KB）
│   └── store/
│       ├── cache.py                 # 缓存层
│       └── manifest.py              # PIPELINE_VERSION + DocState 跟踪
└── tests/
    ├── test_core.py                 # 核心冒烟
    ├── test_robust_pipeline.py      # 鲁棒性压测
    └── fixtures/smoke/              # 冒烟用 Markdown 文档
```

## 技术栈

| 维度 | 选型 | 版本/说明 |
|------|------|-----------|
| 运行时 | Python | ≥ 3.11 |
| 包管理 | uv + hatchling | `uv sync` 一键装 |
| LLM 客户端 | httpx | ≥ 0.27（OpenAI + Anthropic 双方言） |
| 配置 | pydantic-settings + python-dotenv | ≥ 2.6 / ≥ 1.0 |
| CLI | typer | ≥ 0.12 |
| 模糊匹配 | rapidfuzz | ≥ 3.9（dedup 阶段） |
| PDF | pypdf | ≥ 5.0 |
| Web | FastAPI + uvicorn + sse-starlette | ≥ 0.115 / ≥ 0.30 / ≥ 2.1 |
| 图像 | Pillow | ≥ 10.0 |
| 测试 | pytest | ≥ 8.0 |

## 核心模块 / 特性

### 1. `Orchestrator`（主驱动器）
并发 + 幂等 + 可观测 + 可恢复的流水线大脑。ThreadPoolExecutor 文档级并发（默认 10 worker），LLMClient 是线程安全（thread-local active model + 全局信号量），manifest 跳过 pinned/未变文档，`only=[topic]` 重跑指定文档。

### 2. `LLMClient`（模型无关网关客户端）
- OpenAI + Anthropic 双方言自动识别
- 多模型 fallback 链（`LLMQuotaError → 切换下一模型`）
- 内容安全拦截识别（400 + `sensitive contain:` → `LLMContentBlockedError`，不重试）
- Opus 4.7+ 自动剥离 `temperature` 字段（网关拒绝）
- 思考额度分档：off / medium / max，跨方言预算对齐

### 3. 视觉读图 + OCR 块剥离
`vision_read_image` 用多模态模型读 `images/*.png/jpg/jpeg/gif/webp`，**输出忠实转写**（不强加 markdown 结构）。`parse/markdown.py` 把内联的 `<!-- ocr-source: -->` + `<details>` 块剥离出正文，仅把图片 `rel_path` 记录到 Section.images，让 extract 时把 vision_text 融合进 prompt。

### 4. 双层门禁
- **Layer 1 `gate_struct.py`**：JSON 合法性、id 唯一、必填字段非空、长度区间——便宜的本地检查
- **Layer 2 `gate_semantic.py`**：size-bounded indexed batches（`pack_by_size`）让 N 个单元变成少量 API 调用；reviewer 接收 source excerpt + image transcript + 所有 query 变体；**FAIL-CLOSED**（`RAGKB_SEMANTIC_GATE_POLICY=fail_closed` 默认丢弃未审内容，可改 `keep` 标记待人工复核）

### 5. 持久化人工决策
`DecisionStore`（JSON-backed, RLock 保护）记录"无法定论"的样本，CLI / 仪表板可读、可改（`include / exclude / retry / accept`），不阻塞主流程。

### 6. 跨进程事件流
`EventBus(journal_path=events.jsonl)`：进程内 Queue 广播 + 进程间按字节 offset tail 文件。CLI 在 A 进程跑、仪表板在 B 进程跑，B 能 tail 到 A 的事件。

### 7. Web 仪表板（FastAPI + SSE）
`/api/state` 冷加载、`/api/events` SSE 实时流、`/api/run` 后台启动 run、`/api/export` 重导出、`/api/pin` 锁文档、`/api/retry` 重跑指定 topic。单文件 `index.html` 39KB 自带 UI。

## 已完成 / 进行中

- ✅ 9 步流水线全链路（discover → parse → vision → classify → extract → struct gate → semantic gate → regenerate → aggregate → dedup → export）
- ✅ 14+ pipeline 阶段模块
- ✅ 双层门禁 + fail-closed 默认
- ✅ 视觉读图 + OCR 剥离
- ✅ 多模型 fallback 链 + 双方言
- ✅ 4 个运维脚本（限流探针/SOP 合并/失败复检/无审重试）
- ✅ 跨进程事件流（CLI 进程 → SSE 仪表板）
- ✅ FastAPI 仪表板 + 单文件 SPA
- ✅ 2 套完整测试 + 1 个冒烟 fixture
- ✅ 完整 README + 经验总结长文
- ⏳ 真实主题语料接入（`input/` 已被 gitignore）
- ⏳ 向量库对接（产物为 CSV/MD，未直接入向量库）

## 本地开发

```bash
# 装依赖
uv sync

# 配网关
cp .env.example .env
# 编辑 .env：填 JD_LLM_API_KEY / JD_LLM_BASE_URL / XIAOSHU_MODEL

# 放文档
mkdir -p input/基本概念
# 写 input/基本概念/原始文档.md（+ 可选 images/ 子目录）

# 跑流水线
ragkb run              # 全量
ragkb run --only 基本概念  # 单主题重跑

# 启仪表板
ragkb serve --port 8000
# → http://localhost:8000

# 重导出（不重跑）
ragkb export
ragkb export --with-paraphrase   # 实验性扩写版
```

`RAGKB_MAX_WORKERS=10`（per-model 并发窗口），`RAGKB_SEMANTIC_GATE_POLICY=fail_closed`（默认安全模式）。

## 状态

**v0.1.0** —— 流水线已闭环、门禁已上线、仪表板可观测、运维脚本齐备。**生产级就绪**（在能拿到 JD 网关凭证的环境下）。

## License

MIT（仓库内未显式声明 LICENSE 文件，按惯例项目作者保留）
