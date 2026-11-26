"""Tool registry and safe execution. See docs/TOOLS_SPEC.md."""
import ast
import asyncio
import datetime
import logging
import operator
from dataclasses import dataclass
from typing import Callable
from zoneinfo import ZoneInfo

from starlette.concurrency import run_in_threadpool

log = logging.getLogger("llmchat")


@dataclass
class ToolContext:
    """Built by the server from the authenticated session. The model can never supply or change these."""
    user_id: str
    kb: object
    settings: object


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict, ToolContext], str]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def safe_eval(expr: str):
    """Arithmetic only, via an AST whitelist (never eval). Guards against huge exponents and results."""
    if len(expr) > 200:
        raise ValueError("expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow):
                if abs(b) > 1000 or (isinstance(a, int) and isinstance(b, int) and b > 0 and a.bit_length() * b > 4000):
                    raise ValueError("exponent or result too large")
            r = _OPS[type(n.op)](a, b)
            if isinstance(r, int) and r.bit_length() > 4000:
                raise ValueError("result too large")
            return r
        raise ValueError("unsupported expression")

    try:
        return ev(ast.parse(expr, mode="eval"))
    except SyntaxError:
        raise ValueError("invalid expression")


def _calculator(args, ctx):
    return str(safe_eval(args["expression"]))


def _time(args, ctx):
    tz = args.get("timezone", "UTC")
    if tz.upper() == "UTC":
        zone = datetime.timezone.utc
    else:
        try:
            zone = ZoneInfo(tz)
        except Exception:
            raise ValueError(f"unknown timezone '{tz}'")
    return datetime.datetime.now(zone).isoformat(timespec="seconds")


def _search(args, ctx):
    s = ctx.settings
    hits = ctx.kb.search(ctx.user_id, args["query"], s.rag_top_k, s.rag_min_score, s.rag_mode, s.rag_min_bm25)
    return "\n".join(f'[{h["n"]}] ({h["document"]}) {h["text"][:500]}' for h in hits) or "no matching documents"


def default_tools() -> dict[str, Tool]:
    obj = lambda props, req: {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": req}
    tools = [
        Tool("calculator", "Evaluate an arithmetic expression such as '(2+3)*4'. Supports + - * / // % ** and parentheses.",
             obj(["expression"], ["expression"]), _calculator),
        Tool("get_current_time", "Get the current date and time in an IANA timezone such as 'Asia/Kolkata' (default UTC).",
             obj(["timezone"], []), _time),
        Tool("search_documents", "Search the user's uploaded documents and return relevant excerpts.",
             obj(["query"], ["query"]), _search),
    ]
    return {t.name: t for t in tools}


def tool_specs(tools: dict[str, Tool]) -> list[dict]:
    return [{"name": t.name, "description": t.description, "schema": t.schema} for t in tools.values()]


def validate(schema: dict, args) -> str | None:
    """Minimal JSON-schema check: object, required keys, no unknown keys, string types."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    for k in schema.get("required", []):
        if k not in args:
            return f"missing required argument '{k}'"
    for k, v in args.items():
        if k not in props:
            return f"unknown argument '{k}'"
        if props[k].get("type") == "string" and not isinstance(v, str):
            return f"argument '{k}' must be a string"
    return None


async def execute(tools: dict[str, Tool], name: str, args, ctx: ToolContext, timeout: float) -> str:
    """Run one tool call. Always returns text (errors included) so the model can recover."""
    tool = tools.get(name)
    if tool is None:
        return f"error: unknown tool '{name}'"
    problem = validate(tool.schema, args)
    if problem:
        return f"error: {problem}"
    try:
        out = await asyncio.wait_for(run_in_threadpool(tool.fn, args, ctx), timeout)
    except asyncio.TimeoutError:
        return "error: tool timed out"
    except (ValueError, ArithmeticError) as e:
        return f"error: {e}"
    except Exception:
        log.exception("tool %s failed", name)
        return "error: tool failed"
    return str(out)[:2000]
