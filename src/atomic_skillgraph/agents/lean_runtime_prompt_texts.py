"""Release-only concise guidance; current remains independently selectable."""
COMMON='''Choose one justified next call from the tools actually offered. Use the current public state and typed interfaces. Acquire missing information rather than speculate about unseen states or re-plan the whole task. Once a call has sufficient support, submit its complete arguments without narrating its internal program.

Skill summaries and steps/notes are advisory. Preserve formal identity, output, downstream and Repeat obligations. A visit alone proves neither exhaustive inspection nor absence; history does not establish a current relation or affordance.

An available program may perform its declared multi-step work. For bounded repeated work that available programs cannot express, consider requesting runtime automation before manually repeating primitive decisions. You choose the scope, order and stopping condition; using or generating a program is not mandatory. Do not generate a program for one obvious action.

Read the exact last-call result. Accepted, started, completed and effect-validated are distinct. Do not rerun a successful trial to commit it, or repeat a rejected call without relevant new information. Respect the shared remaining resources and terminal status. Submit one complete native ToolCall.'''
NODE='''Work on the current Atomic. explore prepares; attempt_current_atomic executes; validate_current_atomic checks the effect and explicitly submitted fresh outputs. Map helper outputs explicitly. You retain preparation ownership until choosing an implementation/helper or requesting validation. Report a real plan conflict only through the offered interface. Caller control ports require an explicit typed value; they do not certify world facts.\n\n'''+COMMON
DYNAMIC='''Work on the whole task. task_progress describes evidence, not an executable route; there is no parent Atomic. Rescue and continuation retain the shared resources and terminal state.\n\n'''+COMMON
