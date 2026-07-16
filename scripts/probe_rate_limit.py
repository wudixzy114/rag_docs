"""限流探针 — 测试网关对多个模型的限流(429 code:2003)是否 PER-MODEL 独立。

用途：判断「同时并行跑 Gemini-3-Flash / Gemini-3.1-Pro / Claude-Sonnet-4.6 /
Claude-Opus-4.6 / Claude-Opus-4.7 五个模型，把它们的并发数相加」是否可行。
- 若限流是 per-model：五个模型各自有独立的限流桶，并行跑吞吐≈五者之和 → 值得改调度器。
- 若限流是 per-account/gateway：并行只会更快撞同一堵墙，吞吐不翻倍 → 不值得。

方法（两段对照实验）：
  A. 基线：只压 ONE 模型，并发 N，测它单独的 429(2003) 触发率。
  B. 并行：五个模型 EACH 并发 N（总 5N 请求同时打），测每个模型各自的 429 率。
若 B 里每个模型的 429 率 ≈ A（各自基线），说明限流分开；
若 B 里 429 率显著高于 A（尤其是原本不限流的模型也开始 429），说明共享账号级限流。

直连网关发原始请求，NOT 走 client 的重试/退避/fallback —— 要观测网关的
**原始** 429，不能被客户端平滑掉。只发极小的 "ping" 请求，省 token。

运行：  uv run python scripts/probe_rate_limit.py
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx

from ragkb.config import get_llm_settings

# 被测的 5 个模型（全多模态；此处只发文本 ping，够测限流）。
MODELS = [
    "Gemini-3-Flash-Preview-joybuilder",
    "Gemini-3.1-Pro-Preview-joybuilder",
    "Claude-Sonnet-4.6-joybuilder",
    "Claude-Opus-4.6-joybuilder",
    "Claude-Opus-4.7-joybuilder",
]

# 每个模型在每段实验里连发多少请求（并发）。压到能触发限流又不过分烧配额。
REQUESTS_PER_MODEL = 12
# 极小 prompt + 极小输出：只为触发一次网关计数，尽量不耗 token。
PING_MAX_TOKENS = 4


@dataclass
class Outcome:
    ok: int = 0
    rate_limited: int = 0      # 429 code:2003 请求次数超限（我们要测的）
    quota: int = 0             # 429 code:2001 配额耗尽（另一回事，排除）
    other_err: int = 0
    codes: Counter = field(default_factory=Counter)
    latencies: list = field(default_factory=list)


def _classify(status: int, body: str) -> str:
    t = body or ""
    if status == 429 and ("2001" in t or "配额" in t):
        return "quota"
    if status == 429 and ("2003" in t or "限流" in t or "Resource exhausted" in t
                          or "请求次数" in t):
        return "rate_limited"
    if status == 200:
        return "ok"
    return "other_err"


def _one_request(client: httpx.Client, settings, model: str) -> tuple[str, int]:
    """直连网关发一个最小请求，返回 (分类, http状态)。不重试、不退避。"""
    provider = settings.provider_for(model)
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    base = provider.base_url.rstrip("/")

    if provider.dialect == "anthropic":
        url = base + "/messages"
        payload = {"model": model, "max_tokens": PING_MAX_TOKENS,
                   "messages": [{"role": "user", "content": "ping"}]}
        # Opus 4.7+ 拒绝 temperature，这里本就不带，安全。
    elif provider.dialect == "gemini":
        url = base + "/responses"
        payload = {"model": model,
                   "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                   "generationConfig": {"maxOutputTokens": PING_MAX_TOKENS}}
    else:
        url = base + "/chat/completions"
        payload = {"model": model, "max_tokens": PING_MAX_TOKENS,
                   "messages": [{"role": "user", "content": "ping"}]}

    t0 = time.monotonic()
    try:
        r = client.post(url, json=payload, headers=headers)
        dt = time.monotonic() - t0
        cls = _classify(r.status_code, r.text)
        return cls, r.status_code, dt
    except httpx.HTTPError as exc:
        return "other_err", -1, time.monotonic() - t0


def _burst(settings, models: list[str], per_model: int) -> dict[str, Outcome]:
    """对 models 里每个模型各发 per_model 个并发请求，全部同时打出去。"""
    results: dict[str, Outcome] = {m: Outcome() for m in models}
    timeout = httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=15.0)
    with httpx.Client(timeout=timeout) as client:
        tasks = [(m) for m in models for _ in range(per_model)]
        with cf.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futs = {pool.submit(_one_request, client, settings, m): m for m in tasks}
            for fut in cf.as_completed(futs):
                m = futs[fut]
                cls, code, dt = fut.result()
                o = results[m]
                o.codes[code] += 1
                o.latencies.append(dt)
                setattr(o, cls, getattr(o, cls) + 1)
    return results


def _print_table(title: str, res: dict[str, Outcome]) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"{'模型':40s} {'成功':>4s} {'限流':>4s} {'配额':>4s} {'其他':>4s}")
    for m, o in res.items():
        print(f"{m:40s} {o.ok:>4d} {o.rate_limited:>4d} {o.quota:>4d} {o.other_err:>4d}")


def main() -> None:
    settings = get_llm_settings()
    print(f"网关: {settings.base_url}")
    print(f"每模型并发: {REQUESTS_PER_MODEL}  |  被测模型: {len(MODELS)}")

    # 实验 A：逐个模型单独压（基线）。模型之间留 8s 间隔，避免互相干扰。
    print("\n### 实验 A：单模型基线（逐个压，模型间隔 8s） ###")
    baseline: dict[str, Outcome] = {}
    for m in MODELS:
        r = _burst(settings, [m], REQUESTS_PER_MODEL)
        baseline[m] = r[m]
        o = r[m]
        print(f"  {m:40s} 成功{o.ok} 限流{o.rate_limited} 配额{o.quota} 其他{o.other_err}")
        time.sleep(8)

    # 实验 B：五模型同时并行压（总 5×N 请求一齐打）。
    print("\n### 实验 B：五模型并行（总 %d 请求同时打） ###" % (REQUESTS_PER_MODEL * len(MODELS)))
    parallel = _burst(settings, MODELS, REQUESTS_PER_MODEL)

    _print_table("实验 A 基线（各模型单独）", baseline)
    _print_table("实验 B 并行（五模型同时）", parallel)

    # 判定
    print(f"\n{'='*72}\n判定\n{'='*72}")
    verdict_per_model = True
    for m in MODELS:
        a, b = baseline[m], parallel[m]
        a_rl = a.rate_limited
        b_rl = b.rate_limited
        # 若并行时某模型限流明显恶化（且基线几乎不限流），倾向账号级共享限流。
        worse = b_rl > a_rl + max(2, REQUESTS_PER_MODEL // 3)
        flag = "⚠️并行明显恶化" if worse else "≈基线"
        if worse:
            verdict_per_model = False
        print(f"  {m:40s} 基线限流{a_rl:2d} → 并行限流{b_rl:2d}  {flag}")

    print()
    if verdict_per_model:
        print("结论倾向：限流 PER-MODEL 独立 → 五模型并行可提高吞吐，值得改调度器。")
    else:
        print("结论倾向：限流疑似 ACCOUNT/GATEWAY 级共享 → 并行不会线性提吞吐，改调度器收益有限。")
    print("注：这是压力观测的启发式判断，非网关文档保证；如需确证请结合网关团队确认。")


if __name__ == "__main__":
    main()
