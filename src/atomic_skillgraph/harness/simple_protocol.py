"""Public tools and sealed submission; scoring is not a callable tool."""
from dataclasses import dataclass
from typing import Protocol

from ..agents.protocol import validate_schema_instance


@dataclass(frozen=True)
class Capabilities:
    interaction: str = "tool_loop"
    input_modalities: tuple = ("text",)
    tool_surface: str = "named_tools"
    checkpoint_mode: str = "none"
    local_check: str = "unavailable"


class SimpleAdapter(Protocol):
    capabilities: Capabilities

    def reset(self, public_task): ...
    def observe(self): ...
    def available_tools(self): ...
    def call(self, tool_name, arguments): ...
    def submit(self, final_output): ...
    def evaluate(self, sealed_output): ...
    def close(self): ...


class Broker:
    """Shared task counter and the only host bridge exposed to a Program."""
    def __init__(self, adapter, call_limit):
        self.adapter, self.call_limit = adapter, call_limit
        self.events = []
        self.done = False

    def observe(self):
        return self.adapter.observe()

    def available_tools(self):
        return [] if self.done else self.adapter.available_tools()

    def remaining_calls(self):
        return 0 if self.done else max(0, self.call_limit - len(self.events))

    def call(self, name, arguments):
        if self.remaining_calls() <= 0:
            raise RuntimeError("Task tool budget exhausted or environment terminated")
        specs = {tool["name"]: tool for tool in self.available_tools()}
        if name not in specs:
            return {"accepted": False, "status": "blocked", "error": "tool is not currently available",
                    "observation": self.observe()}
        validate_schema_instance(arguments, specs[name]["input_schema"])
        # Reserve the native call before execution, including rejected calls.
        event = {"name": name, "arguments": arguments, "index": len(self.events)}
        self.events.append(event)
        try:
            result = self.adapter.call(name, arguments)
        except Exception:
            event["execution_error"] = True
            raise
        event["result"] = result
        self.done = bool(result.get("done", False))
        return result

    def check_local(self, inputs, outputs, start):
        check = getattr(self.adapter, "check_local", None)
        return check(inputs, outputs, self.events[start:]) if check else "unavailable"
