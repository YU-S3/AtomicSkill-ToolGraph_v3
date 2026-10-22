"""Read-only completion audit and combined Test1400 statistics."""
import json
import continue_kimi as run

r, c = run.r, run.c
c.check(run.STAGE)
stage = run.ROOT / run.STAGE
result = r.read(stage / 'result.json')
assert result['status'] == 'completed' and result['error'] is None
assert all(r.read(stage / 'freeze_audit.json').values())
snapshot = r.read(run.HERE / 'parent_snapshot.json')
parent = c.completed(run.PARENT / run.STAGE, run.ARM)
child = c.completed(stage, run.ARM)
assert len(parent) == 1390 and len(child) == 10 and not set(parent) & set(child)
assert r.digest(parent) == snapshot['result_digest']
assert r.sha(run.PARENT / 'budget.json') == snapshot['budget_sha256']
assert r.sha(run.PARENT.parent / 'glm53_flash' / run.STAGE / 'result.json') == snapshot['glm_result_sha256']
rows = {**parent, **child}
tasks = r.read(run.PARENT / run.STAGE / 'tasks.json')['tasks']
assert set(rows) == {t['task_id'] for t in tasks} and len(rows) == 1400
ledger, inherited = r.read(run.ROOT / 'budget.json'), r.read(run.ROOT / 'inherited_budget.json')
assert ledger['limits'] == inherited['limits'] == {'token_limit': 8000000, 'request_limit': 3000, 'finish_reserve': 0}
assert all(ledger['attempts'][k] == v for k, v in inherited['attempts'].items())
refused = snapshot['refused_attempt']
refused_id = refused['request_context']['parent_task_id']
assert child[refused_id]['score']['hard'] is False and child[refused_id]['execution']['prediction'] is None
assert child[refused_id]['requests'][0]['http_attempts'] == [refused]
assert child[refused_id]['usage'] == []
added = {k: v for k, v in ledger['attempts'].items() if k not in inherited['attempts']}
assert len(added) == 9 and all(v['status'] == 'metered' for v in added.values())
assert {v['context']['parent_task_id'] for v in added.values()} == set(child) - {refused_id}
all_request_ids = [a['request_id'] for row in rows.values() for q in row['requests'] for a in q['http_attempts']]
assert len(all_request_ids) == len(set(all_request_ids)) == len(ledger['attempts']) == 1400
assert set(all_request_ids) == set(ledger['attempts'])
tokens = sum(u['total_tokens'] for row in rows.values() for u in row['usage'])
assert tokens == sum(v.get('accounted_tokens', 0) for v in ledger['attempts'].values())
unknown = {k: v for k, v in ledger['attempts'].items() if v['status'] != 'metered'}
assert set(unknown) == {refused['request_id']}
assert unknown[refused['request_id']]['reserved_tokens'] == 77429
assert 'accounted_tokens' not in unknown[refused['request_id']]
calls = [a for row in rows.values() for a in row['execution'].get('attempts', [])]
assert len(calls) == 1399 and all(a['status'] == 'ok' and a['outputs_consumed'] for a in calls)
summary = {
    'passed': True, 'at': r.utc(), 'status': 'completed', 'model': 'kimi-k2.6',
    'bank': 'human.searchqa.accuracy.v7.text_compiler', 'n': 1400,
    'correct': sum(bool(row['score']['hard']) for row in rows.values()),
    'EM_pct': sum(bool(row['score']['hard']) for row in rows.values()) / 14,
    'F1_pct': sum(row['score']['soft'] for row in rows.values()) / 14,
    'tokens': tokens, 'prompt_tokens': sum(u.get('prompt_tokens', 0) for row in rows.values() for u in row['usage']),
    'completion_tokens': sum(u.get('completion_tokens', 0) for row in rows.values() for u in row['usage']),
    'physical_http': len(ledger['attempts']), 'unknown_usage_requests': 1,
    'unknown_reserved_tokens': 77429, 'admission_charge_including_reservation': tokens + 77429,
    'provider_refusals_scored_in_denominator': 1, 'program_ok': len(calls),
    'new_http_in_continuation': len(added), 'new_tokens_in_continuation': sum(v['accounted_tokens'] for v in added.values()),
    'original_1390_results_and_costs_preserved': True, 'no_refusal_retry': True,
    'source_model_bank_input_and_budget_unchanged': True, 'glm_results_unchanged': True,
    'refusal_adapter_version': run.outcomes.VERSION, 'refusal_adapter_sha256': r.sha(run.outcomes.__file__),
    'parent': str(run.PARENT), 'continuation': str(run.ROOT), 'source': r.code_identity(),
    'real_http_from_audit': 0}
r.write_json(run.HERE / 'final_summary.json', summary)
r.write_json(run.ROOT / 'combined_summary.json', summary)
r.write_json(run.HERE / 'test1400_host_only.json', [{
    'task_id': t['task_id'], 'order_index': i, 'prediction': rows[t['task_id']]['execution']['prediction'],
    'score': rows[t['task_id']]['score'], 'reason': rows[t['task_id']]['execution']['reason'],
    'tokens': sum(u['total_tokens'] for u in rows[t['task_id']]['usage']),
    'origin': 'parent' if t['task_id'] in parent else 'continuation'} for i, t in enumerate(tasks)])
report = f'''# Kimi V7 Test1400 完成核验

Kimi-K2.6 完整测试已完成：**{summary['correct']}/1400，EM {summary['EM_pct']:.2f}%，F1 {summary['F1_pct']:.2f}%**。

已计量 {tokens:,} tokens，物理请求 1400 次。另有一次服务端拒答未返回用量，保留 77,429 tokens 预留；含预留的额度占用为 {tokens + 77429:,}，未超过原 8,000,000 tokens／3000 HTTP 上限。

旧 1390 题结果、原始请求及账本保持原样。第 1391 题的原请求收到 Moonshot HTTP 400，结构化错误类型为 `content_filter`。新执行适配器据此按空答评分，未重发该题，未伪造 finish_reason 或实际用量。其余九题使用原模型配置、公开输入、题序及 V7 Bank 完成，共新增 {summary['new_tokens_in_continuation']:,} tokens／9 HTTP。

六项零 HTTP 生产链路回归及五项拒答分类反例通过。非拒答参数错误、服务器错误和断流仍停止，不写成普通答错。1399 次 Program 执行及输出消费全部成功；拒答题没有伪造 Program 执行。Frozen、原源码、模型及方法配置未变，适配器版本单独记录为 `{run.outcomes.VERSION}`。

原运行：`{run.PARENT}`

续跑目录：`{run.ROOT}`

`final_summary.json` 为合并后的最终结果，`test1400_host_only.json` 为按 canonical 顺序的逐题表；旧父 run 继续保留 stopped 状态以反映实际历史。GLM 完成结果未修改。
'''
(run.HERE / '完成核验报告.md').write_text(report, encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
