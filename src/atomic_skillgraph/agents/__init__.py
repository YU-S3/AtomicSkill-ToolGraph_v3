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
from .provider_probe import (
    ProviderCapabilityError,
    ensure_provider_capability,
    run_provider_capability_probe,
)
from .session import ClientManagedAgentSession, ProtocolFailureRecord, ReplayAgentSession
from .structured_submission import StructuredSubmission, StructuredSubmissionClient
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
    "ProviderCapabilityError",
    "REAL_USAGE_BUCKETS",
    "ReplayAgentSession",
    "SchemaValidationError",
    "StructuredSubmission",
    "StructuredSubmissionClient",
    "UsageBucket",
    "UsageEvent",
    "UsageLedger",
    "sum_usage",
    "ensure_provider_capability",
    "parse_json_strict",
    "validate_schema_instance",
    "run_provider_capability_probe",
]
