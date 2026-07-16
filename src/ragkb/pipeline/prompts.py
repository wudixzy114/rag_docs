"""All LLM prompts, centralized.

Each prompt group carries a VERSION. Cache keys include the relevant version, so
editing a prompt here automatically invalidates only that stage's cache on the
next run — no manual cache clearing, no stale outputs from an old prompt.

Prompts are written to work on a WEAK model too (Sonnet / DeepSeek-Pro): explicit
output contracts, JSON-only where parsed, no reliance on the model volunteering
structure. That's the "低质量模型也能过" acceptance bar — the deterministic gates
downstream assume the prompt did its best to be parseable.
"""
from __future__ import annotations

# ---------------------------------------------------------------- vision -----
VISION_VERSION = "v1"

VISION_SYSTEM = """你是一个严谨的技术图片分析助手，服务于一个运维/训练平台的诊断知识库。\
图片多为：报错截图、日志片段、资源监控图表、控制台/UI 面板、配置界面。\
你的转写将作为知识库的事实来源，必须精确、忠实，绝不臆造。"""

# The inline OCR is passed in as a low-confidence hint. The model's own vision is
# authoritative; the hint only helps disambiguate garbled characters.
VISION_USER = """请分析这张图片，输出两部分（用给定标记分隔）：

[TRANSCRIPT]
忠实转写图片中的所有文字（中英文、数字、符号、路径、命令、报错信息）。\
保留原有的结构：表格用 Markdown 表格，列表用列表，日志/代码保留原样换行。\
不要翻译，不要改写，不要补全你看不清的内容——看不清就写 [模糊]。

[MEANING]
用 1-3 句话说明这张图在诊断语境下表达了什么（例如：这是一张内存监控图，\
显示 worker-0 内存在 14:00 达到 69.5GiB 后骤降为 0，符合 OOM 特征）。\
只根据图片内容陈述，不要推测图片以外的信息。

已有一份低质量 OCR 参考（可能有错，仅供辨认模糊字符时参考，不要盲信）：
<ocr_hint>
{ocr_hint}
</ocr_hint>"""


def build_vision_user(ocr_hint: str) -> str:
    hint = (ocr_hint or "").strip() or "（无）"
    return VISION_USER.format(ocr_hint=hint)


# ------------------------------------------------------------- classify -----
CLASSIFY_VERSION = "v2"

CLASSIFY_SYSTEM = """你是知识库内容分类助手。给定一个文档小节（标题+正文摘要），\
判断它更适合作为哪一类知识单元。

默认倾向 sop——只有**明确呈现出问答结构**的内容才归 qa。分类标准：

- "qa"：**必须同时具备**「一个具体的问题/现象/报错」和「针对它的原因分析、定位方法或解决办法」，\
  能被一个用户问题直接命中。典型：`任务一直 OOM 怎么排查？`、`报错 xxx 是什么原因`、\
  `为什么任务排队半天不动`。**判定要严**：如果小节只是在"介绍/说明/公告"某个东西，\
  而不是在"解答一个故障或疑问"，就不是 qa。
- "sop"：流程/步骤/规则/说明/介绍类内容，以及一切**说明性而非问答性**的内容。典型特征：\
  有序的操作步骤、排查流程、规则说明、概念介绍、名词解释；\
  **尤其包括**产品动态、版本更新记录、发布公告、功能上线说明、更新日志（如标题是日期 \
  `2023.11.15` 或版本号 `V2.5.0`，正文在陈述"新增/上线/升级了什么功能"）——\
  这些都归 sop，绝不归 qa。强行拆成问答会丢失上下文或产生空洞问答。
- "skip"：无实质知识价值的内容。如纯目录、单纯的分隔标题、空段落、\
  "提问之前"之类的引导语。

判断口诀：**在"解答故障/疑问" → qa；在"介绍/说明/公告什么" → sop。拿不准时归 sop。**

只输出 JSON，不要解释。"""

CLASSIFY_USER = """对下列小节逐个分类。每个小节给出 id、标题、正文摘要。

返回一个 JSON 数组，每个元素：{{"id": <数字>, "label": "qa|sop|skip", "reason": "<不超过20字>"}}
必须覆盖所有 id，顺序不限。

小节列表：
{sections}"""


def build_classify_user(sections: list[dict]) -> str:
    lines = []
    for s in sections:
        body = (s.get("body_preview") or "").replace("\n", " ")[:300]
        lines.append(f'--- id={s["id"]}\n标题: {s["title"]}\n正文摘要: {body}')
    return CLASSIFY_USER.format(sections="\n".join(lines))


# -------------------------------------------------------------- extract -----
EXTRACT_VERSION = "v2"

EXTRACT_SYSTEM = """你是诊断知识库的问答抽取专家。你的产出将直接进入生产环境的向量库，\
供一线用户检索。质量要求（工业级）：

1. 精准：answer 必须是明确、可执行、忠于原文的表述，禁止"可能需要检查一下相关配置"\
   这类模糊话。原文有具体命令、版本号、路径、参数，必须原样保留。
2. 完整：answer 要自成一体，用户不看原文也能照做。若原文含"如何定位/如何解决"分步，\
   要完整纳入，不能截断。
3. 忠实：只使用给定材料（正文 + 图片转写），绝不编造。材料没提到的不要写。
4. 图文融合：正文里引用的截图，其转写内容已提供，要把其中的关键信息（报错文本、\
   监控数值、界面字段）自然融入 answer。
5. query 用用户口吻的自然问题（"任务 OOM 了怎么排查？"），不要用文档标题的编号腔调。

只输出 JSON，不要解释。"""

EXTRACT_USER = """从下面这个小节抽取一个或多个高质量问答对。多数小节抽 1 个即可；\
若明显覆盖多个独立问题，可抽多个。

返回 JSON 数组，每个元素：
{{"query": "<用户口吻的问题>",
  "answer": "<精准、完整、可执行的回答，保留命令/版本/路径等原文细节>",
  "grounded": true}}

若这个小节无法产出有价值的问答（内容太空、纯标题），返回空数组 []。

小节标题路径：{heading_path}
小节标题：{title}

正文：
{body}

{images_block}"""


def build_extract_user(heading_path: str, title: str, body: str,
                       images_block: str) -> str:
    return EXTRACT_USER.format(heading_path=heading_path or "（无）", title=title,
                               body=body or "（无正文）",
                               images_block=images_block or "（本小节无图片）")


BATCH_EXTRACT_USER = """一次处理下列多个互相独立的小节，以减少调用成本。不得跨小节混合事实。

返回 JSON 数组，必须覆盖每个 id：
[{{"id":"<原 id>","items":[{{"query":"<问题>","answer":"<忠于该小节的完整回答>"}}]}}]
无法产出问答的小节也要返回 {{"id":"<原 id>","items":[]}}。

小节：
{sections}"""


def build_batch_extract_user(sections: list[dict]) -> str:
    import json
    return BATCH_EXTRACT_USER.format(
        sections=json.dumps(sections, ensure_ascii=False, separators=(",", ":")))


# ------------------------------------------------------------------ sop -----
SOP_VERSION = "v1"

SOP_SYSTEM = """你是诊断知识库的流程文档整编专家。给定一个流程/说明类小节，\
你要产出：

1. 一篇清洗后的、结构良好的 Markdown 文档：保留完整的步骤顺序和逻辑，\
   把图片转写中的关键信息（命令、报错、界面）自然融入正文，去掉 OCR 噪声和无关碎片。\
   单一职责——只讲这一个流程/主题，不要混入无关内容。禁止截断、禁止臆造。
2. 3-5 条"用户口吻的入口问题"：用户在遇到相关问题时可能会问的话，用来把用户的\
   现象查询路由到这篇流程。要贴近真实用户话术（"任务一直排队是为什么？"），\
   覆盖不同问法/别名。

只输出 JSON，不要解释。"""

SOP_USER = """整编下面这个流程/说明类小节。

返回 JSON 对象：
{{"markdown": "<清洗后的完整 Markdown，含标题>",
  "entry_questions": ["<入口问题1>", "<入口问题2>", ...]}}

小节标题路径：{heading_path}
小节标题：{title}

正文：
{body}

{images_block}"""


def build_sop_user(heading_path: str, title: str, body: str,
                   images_block: str) -> str:
    return SOP_USER.format(heading_path=heading_path or "（无）", title=title,
                           body=body or "（无正文）",
                           images_block=images_block or "（本小节无图片）")


def build_images_block(images: list[dict]) -> str:
    """Render image transcriptions for injection into extract/sop prompts.
    `images` = [{rel_path, transcript, meaning}]. Empty → "" (caller shows a
    'no images' placeholder)."""
    if not images:
        return ""
    parts = ["【本小节引用的图片（已由视觉模型转写）】"]
    for im in images:
        parts.append(f"- 图片 {im['rel_path']}:")
        if im.get("transcript"):
            parts.append(f"  转写:\n{_indent(im['transcript'], 4)}")
        if im.get("meaning"):
            parts.append(f"  含义: {im['meaning']}")
    return "\n".join(parts)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln for ln in (text or "").splitlines())


# ----------------------------------------------------------- review (L2) -----
REVIEW_VERSION = "v2"

REVIEW_SYSTEM = """你是诊断知识库的质量审核专家，用独立、挑剔的眼光审查已抽取的问答对是否\
达到生产标准。你不是来挑语法毛病的，而是判断这条问答能否安全地交给一线用户使用。

对每条问答，检查：
1. 准确性：answer 是否与提供的原始材料一致，有无编造、张冠李戴。
2. 明确性：有无"可能""也许""检查一下相关设置"这类模糊、不可执行的表述。
3. 完整性：answer 是否自成一体、步骤/命令完整，没有被截断或缺关键步骤。
4. 相关性：query 和 answer 是否匹配，query 是否是真实用户会问的问题。
5. 扩写问法：逐条检查问法变体是否与主 query 意图完全一致；引入原文没有的新故障场景、
   前提或结论的变体必须排除。

判定 verdict：
- "pass"：达到生产标准，可直接使用。
- "revise"：基本正确但有可修复的小问题（模糊表述、query 措辞不佳等）。
- "reject"：有事实错误、编造、严重不完整或答非所问。

只输出 JSON 数组，不要解释。"""

REVIEW_USER = """审核下列问答对。每条给出 id、query、answer，以及抽取时依据的原始材料。

返回 JSON 数组，每个元素：
{{"id": <数字>, "verdict": "pass|revise|reject", "reason": "<不超过40字>",
  "valid_paraphrases": ["<从输入问法变体中逐字复制审核通过的项；不得新增>"]}}
必须覆盖所有 id。

{items}"""


def build_review_user(items: list[dict]) -> str:
    blocks = []
    for it in items:
        src = (it.get("source") or "").strip()[:6000]
        blocks.append(
            f'--- id={it["id"]}\n'
            f'query: {it["query"]}\n'
            f'answer: {it["answer"]}\n'
            f'问法变体: {it.get("paraphrases", [])}\n'
            f'原始材料:\n{src}')
    return REVIEW_USER.format(items="\n\n".join(blocks))


# ------------------------------------------------------- SOP review (L2) ----
SOP_REVIEW_VERSION = "v1"

SOP_REVIEW_SYSTEM = """你是诊断知识库的流程文档质量审核专家。请逐篇对照原始材料审核 SOP，重点检查：

1. 忠实性：正文中的事实、命令、参数、版本、路径和结论必须有原文依据，不能张冠李戴或编造。
2. 完整性：原文中的必要步骤、先后顺序、前置条件、分支、例外和警告不得遗漏，也不能截断。
3. 可执行性：步骤表述明确，代码块和 Markdown 结构完整，读者无需返回原文即可执行。
4. 入口问题：每条都必须能由该 SOP 回答，不能引入原文没有的新故障场景或结论。

判定 verdict：
- "pass"：可直接发布。
- "revise"：主体有依据，但存在可修复的遗漏、含混或入口问题偏差。
- "reject"：有编造、张冠李戴、严重缺步骤或答非所问。

只输出 JSON 数组，不要解释。"""

SOP_REVIEW_USER = """审核下列 SOP。返回 JSON 数组，必须覆盖所有 id：
{{"id": <数字>, "verdict": "pass|revise|reject", "reason": "<不超过40字>",
  "valid_entry_questions": ["<从输入入口问题中逐字复制审核通过的项；不得新增>"]}}

{items}"""


def build_sop_review_user(items: list[dict]) -> str:
    blocks = []
    for it in items:
        source = (it.get("source") or "").strip()[:16000]
        markdown = (it.get("markdown") or "").strip()[:20000]
        blocks.append(
            f'--- id={it["id"]}\n'
            f'title: {it["title"]}\n'
            f'entry_questions: {it.get("entry_questions", [])}\n'
            f'SOP:\n{markdown}\n'
            f'原始材料:\n{source}')
    return SOP_REVIEW_USER.format(items="\n\n".join(blocks))


# ------------------------------------------------------- paraphrase keys -----
PARAPHRASE_VERSION = "v2"

PARAPHRASE_SYSTEM = """你是检索优化助手。给定一个标准问题，生成若干"用户可能会用的其他问法"，\
用于扩大向量检索的命中面。要求：

- 覆盖不同措辞、口语化说法、同义词、常见别名和缩写。
- 覆盖用户可能只描述"现象"而非"术语"的说法（例如原问题含"OOM"，可加"任务突然被杀掉了""内存爆了"）。
- 保持与原问题相同的意图，不要改变要问的东西，不要编造新场景。
- 每条都是完整、独立的问句。

只输出 JSON 数组（字符串数组），不要解释。"""

PARAPHRASE_USER = """为下面的问题生成 {n} 条不同的用户问法（不含原问题本身）。

返回 JSON：["问法1", "问法2", ...]

原问题：{query}
（供参考的答案要点）：{answer_hint}"""


def build_paraphrase_user(query: str, answer: str, n: int = 4) -> str:
    return PARAPHRASE_USER.format(n=n, query=query,
                                  answer_hint=(answer or "")[:200])


def build_batch_paraphrase_user(items: list[dict], n: int) -> str:
    import json
    return (f"为每条问答生成最多 {n} 个自然且语义等价的用户问法。"
            "只输出 JSON 数组并覆盖所有 id，格式："
            '[{"id":0,"variants":["问法1","问法2"]}]。\n数据：'
            + json.dumps(items, ensure_ascii=False, separators=(",", ":")))


# --------------------------------------------------------- regeneration -----
REGENERATE_VERSION = "v1"

REGENERATE_SYSTEM = """你是知识库问答修复专家。上一版问答未通过审核，你必须仅依据给定原始材料
重新生成。不得为了通过审核而删减原文中的必要步骤、条件、命令、数值或例外；原文覆盖多个必要
环节时必须完整保留。禁止引入材料外事实。只输出 JSON。"""


def build_regenerate_user(items: list[dict]) -> str:
    import json
    return ("逐条修复并覆盖所有 id。返回 JSON 数组："
            '[{"id":0,"query":"用户问题","answer":"完整且忠于原文的回答"}]。\n'
            "输入：" + json.dumps(items, ensure_ascii=False, separators=(",", ":")))


SOP_REGENERATE_VERSION = "v1"

SOP_REGENERATE_SYSTEM = """你是知识库 SOP 修复专家。上一版 SOP 未通过审核，你必须仅依据给定原始材料重新整编。
完整保留必要步骤、顺序、条件、分支、例外、警告、命令、参数、版本和路径；禁止引入材料外事实。
入口问题只能描述该材料确实能回答的意图。只输出 JSON。"""


def build_sop_regenerate_user(items: list[dict]) -> str:
    import json
    return ("逐篇修复并覆盖所有 id。返回 JSON 数组："
            '[{"id":0,"markdown":"含标题的完整 Markdown",'
            '"entry_questions":["用户问题1","用户问题2"]}]。\n'
            "输入：" + json.dumps(items, ensure_ascii=False, separators=(",", ":")))
