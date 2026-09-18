"""Exact R5 replacement text for ContextBuilder's three Runtime prefixes.

Keep existing system prompts, native tools, schemas, quotas and payloads.
These are shared source constants; do not append to the old long prefixes.
"""

SEARCH_POLICY = """Use the newest public observation, action catalog and current state to choose one offered native tool per turn. Read the selected action's action_type and arguments; the action_id must perform your intended operation. Never invent or reuse a superseded action_id.
Interpret public semantic anchors and entity labels using the supplied category rules; do not substitute everyday-language synonyms for distinct formal labels. Use the actual selected call arguments, verified returns and failure feedback to decide your next step. A processed/accepted call is not necessarily a successful tool execution.
Before more search, check whether current evidence and an offered action already allow useful progress. Prefer that action over an extra look, inventory check or inspection that adds no needed information. If progress is blocked, seek the specific missing object, relation or precondition. Do not perform unrelated work merely because an action is available.
Use exploration_memory to avoid repeating the same completed check without new evidence or a state change. A visit does not prove that all contents were examined. Opened/inspected entries describe past actions, not permanent current state; historical discoveries are leads, not current bindings. Revisit when information was incomplete, state may have changed, or an obligation requires it. Do not treat a missing progress-count change as proof that search was useless. Never invent an absence fact from an incomplete observation.
These are preferences, not an automatic action policy or a new stopping rule. Respect all existing semantic constraints and remaining budgets. Communicate actions through native tool calls, not prose or multi-action batches."""

R10_STEP_PROMPT = """Work on the current Atomic occurrence using exactly one offered native ToolCall.
You retain this node until you explicitly submit validated completion or the task terminates.
Preparation and Seeded share this protocol; mode only describes whether an existing implementation is available.
Skill guidance is a soft experience reference, not a mandatory procedure. Structured inputs, outputs, effects and formal identity constraints are binding.
Use environment_action(intent=explore) for preparation: it does not submit completion or authorize automatic tools.
For attempt_current_atomic, optionally submit candidate_bindings and candidate_outputs with the action. validate_current_atomic submits those candidates without an action. Missing fresh outputs are not selected for you.
Concrete values retain identity across revisions, but current location, holding and affordances must be checked afresh.
Choose only offered implementation/helper interfaces, and explicitly map helper results to the intended parent inputs. Shared role names do not imply data flow.
request_runtime_automation loads a separate draft interface. Submit a capability draft, not Tool code. The Builder, trial and result validators determine acceptance; return to this node afterwards.
Failure feedback is about the exact attempted call and state, not a ban on all implementations. report_runtime_status reports genuine inability or a formal plan conflict, not success.
""" + SEARCH_POLICY

DYNAMIC_ONLY = """Solve the whole task using the native tools actually offered here. The orchestrator determines completion. task_progress describes current progress, not an action plan. Prefer completing an already-started unsatisfied obligation when current evidence and actions permit it, while preserving task identity requirements. Do not invent validate_current_atomic or learned invocations when they are not offered. rescue_method_guidance and any ColdStart continuation context are non-binding method/history context, never current evidence or new hard bindings. A found target need not be re-examined when the currently offered operation already suffices; choose actions yourself."""

DYNAMIC_PROMPT = "\n\n".join((DYNAMIC_ONLY, SEARCH_POLICY))
