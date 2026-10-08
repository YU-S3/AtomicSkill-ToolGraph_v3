"""One terminal episode per physical task/attempt, with append-only revisions."""
def terminal_episode(task, trace, ended_at):
    execution = trace['execution']
    return {**task, 'task_end_time': ended_at, 'official_score': trace['score']['raw_score'],
        'success': trace['score']['hard'], 'terminal_reason': execution['reason'],
        'solve_status': trace.get('solve_status', 'completed'),
        'learning_status': trace.get('learning_status', 'not_started'), 'learning_error': trace.get('learning_error'),
        'submission_producer_attempt_id': execution.get('submission_producer_attempt_id'),
        **{k:execution.get(k) for k in ('provider_finish_reason','answer_status','empty_answer','completion_truncated')}}


def episode_projection(rows):
    events, projection = {}, {}
    for row in rows:
        events[row['event_id']] = row
        if row.get('episode_event_type') == 'error': continue
        key = (row['task_id'], row['attempt_id'])
        supersedes = row.get('supersedes_event_id')
        if supersedes and supersedes not in events:
            raise ValueError('Episode revision source is missing')
        if row.get('success') is not None:
            projection[key] = row
    return list(projection.values())
