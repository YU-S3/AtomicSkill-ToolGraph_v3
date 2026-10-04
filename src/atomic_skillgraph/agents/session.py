"""Small shared response-order helpers; task response recovery is durable in empirical."""
import json
PROTOCOL_REPAIR_LIMIT = 1


def repair_messages(turn, error):
    assistant = dict(turn.replay_assistant_message)
    if not assistant or not turn.tool_calls:
        return []
    return [assistant, *[{'role': 'tool', 'tool_call_id': call.call_id,
                         'content': json.dumps({'error': str(error)})} for call in turn.tool_calls]]
