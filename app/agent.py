"""The tool-calling loop: ask the model, run the tools it requests, feed results back, repeat (bounded)."""
from .tools import execute, tool_specs


async def run_tool_loop(provider, turns, system, tools, ctx, max_steps, max_tokens, timeout):
    """Async generator of events: tool_call, tool_result, then one final {"type": "final", "text": ...}."""
    specs = tool_specs(tools)
    for step in range(max_steps + 1):  # max_steps tool rounds, plus one last model call that must be the answer
        res = await provider.step(turns, system, specs, max_tokens)
        if not res["tool_calls"]:
            yield {"type": "final", "text": res["text"]}
            return
        if step == max_steps:  # still asking for tools after the allowed rounds: stop without running them
            break
        calls = res["tool_calls"][:4]  # cap parallel calls per round
        turns.append({"role": "assistant", "content": res["text"], "tool_calls": calls})
        for c in calls:
            yield {"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": c["arguments"]}
            out = await execute(tools, c["name"], c["arguments"], ctx, timeout)
            yield {"type": "tool_result", "id": c["id"], "name": c["name"], "output": out}
            turns.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out})
    yield {"type": "final", "text": "I stopped because the tool-step limit was reached before I could finish."}
