"""Modern provider-independent Agent sessions and usage accounting."""

from .context_builder import ContextBuilder
from .protocol import (
    AgentMessage,
    AgentProvider,
    AgentSession,
    AgentTurn,
    NativeToolCall,
    NativeToolSpec,
    SchemaValidationError,
    parse_json_strict,
    validate_schema_instance,
)
from .provider import (
    AgentProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ProviderProtocolError,
)
from .session import ClientManagedAgentSession, ProtocolFailureRecord, ReplayAgentSession
from .usage import (
    AgentBudget,
    BudgetTracker,
    LLMUsage,
    REAL_USAGE_BUCKETS,
    UsageBucket,
    UsageEvent,
    UsageLedger,
    sum_usage,
)

__all__ = [
    "AgentBudget",
    "AgentMessage",
    "AgentProvider",
    "AgentProviderError",
    "AgentSession",
    "AgentTurn",
    "BudgetTracker",
    "ClientManagedAgentSession",
    "ContextBuilder",
    "LLMUsage",
    "NativeToolCall",
    "NativeToolSpec",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "ProtocolFailureRecord",
    "ProviderProtocolError",
    "REAL_USAGE_BUCKETS",
    "ReplayAgentSession",
    "SchemaValidationError",
    "UsageBucket",
    "UsageEvent",
    "UsageLedger",
    "sum_usage",
    "parse_json_strict",
    "validate_schema_instance",
]
