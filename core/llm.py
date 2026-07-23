"""One OpenAI-compatible client for z.ai: retry, token/cost metering, ledger attribution."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from .ledger import BudgetGuard, Ledger

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def load_env(path: Path = _ROOT / ".env") -> None:
    """Tiny .env loader (KEY=VALUE lines); no python-dotenv dependency."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def load_prices(path: Path = _ROOT / "prices.json") -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class LLM:
    def __init__(self, ledger: Ledger, guard: BudgetGuard, prices: dict | None = None):
        load_env()
        key = os.environ.get("ZAI_API_KEY")
        if not key:
            raise RuntimeError("ZAI_API_KEY not set (env or .env)")
        self.client = OpenAI(
            api_key=key, base_url=os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL)
        )
        self.ledger, self.guard = ledger, guard
        self.prices = prices if prices is not None else load_prices()

    def _cost(self, model: str, p_tok: int, c_tok: int) -> float | None:
        p = self.prices.get(model)
        if not p:
            return None  # unknown price -> guard falls back to max_calls
        return (p_tok * p["in"] + c_tok * p["out"]) / 1e6

    def _meter(self, resp, model: str, role: str, t0: float) -> None:
        u = resp.usage
        usd = self._cost(model, u.prompt_tokens, u.completion_tokens)
        self.guard.charge(usd)
        self.ledger.append({
            "type": "llm_call", "role": role, "model": model,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "usd": usd, "seconds": round(time.monotonic() - t0, 2),
            "text": resp.choices[0].message.content or "",
        })

    def _once(self, model: str, messages: list[dict], temperature: float, role: str,
              max_tokens: int, tools: list | None):
        """One API call with retry/backoff on transient errors; metered."""
        last_err: Exception | None = None
        for attempt in range(3):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, **({"tools": tools} if tools else {}),
                    # GLM thinking mode burns 10x completion tokens and sometimes
                    # spirals past any cap, leaving content empty. Evolution wants
                    # many cheap diverse samples; selection does the deliberating.
                    extra_body={"thinking": {"type": "disabled"}},
                )
                self._meter(resp, model, role, t0)
                return resp.choices[0].message
            except Exception as e:  # 429/5xx/network — retry with backoff
                last_err = e
                time.sleep(2**attempt)
        self.guard.charge(None)
        self.ledger.append({"type": "llm_error", "role": role, "model": model, "error": str(last_err)})
        raise last_err

    def chat(self, model: str, messages: list[dict], temperature: float, role: str,
             max_tokens: int = 16384, tools: list | None = None, tool_handler=None) -> str:
        """One metered chat exchange. `role` (writer/reflect/judge) is for ledger attribution.
        With `tools`, runs a bounded tool loop (max 3 rounds) using `tool_handler(name, args)`."""
        self.guard.check()
        messages = list(messages)
        for round_ in range(3):
            use_tools = tools if (tools and round_ < 2) else None
            msg = self._once(model, messages, temperature, role, max_tokens, use_tools)
            if not (use_tools and msg.tool_calls):
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                result = tool_handler(tc.function.name, json.loads(tc.function.arguments or "{}"))
                self.ledger.append({
                    "type": "tool_call", "role": role, "name": tc.function.name,
                    "args": (tc.function.arguments or "")[:300], "result_chars": len(result),
                })
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return msg.content or ""


def parse_code(text: str) -> str | None:
    """Extract the last fenced code block, any language tag (writers may think
    aloud first). A final block left unterminated by max_tokens truncation counts."""
    blocks = re.findall(r"```[\w+-]*[ \t]*\n(.*?)(?:```|\Z)", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def parse_reasoning(text: str) -> str:
    """Everything before the final code block: the writer's Idea line + rationale."""
    fences = list(re.finditer(r"```[\w+-]*[ \t]*\n", text))
    return (text[: fences[-1].start()] if fences else text).strip()


def parse_idea(text: str) -> str | None:
    """The one-line `Idea:` declaration the writer contract requires."""
    m = re.search(r"^\s*\**Idea\**\s*[:*]\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip().strip("*") if m else None


def parse_prediction(text: str) -> str | None:
    """The one-line `Prediction:` declaration: expected effect + mechanism."""
    m = re.search(r"^\s*\**Prediction\**\s*[:*]\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip().strip("*") if m else None
