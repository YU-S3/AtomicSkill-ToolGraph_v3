# Office HumanStrong V15

Native human-authored Skill → Implementation → Program reference Bank. This is a development-exposed diagnostic asset, not a naturally learned Bank or blind Test result.

The Bank contains 37 scoped guidance atoms and 5 linked Programs. At most 6 atoms are selected from the public question and supplied source text. The Agent chooses native actions; selection does not dispatch a fixed workflow. No task IDs, gold answers, or answer lookup tables enter the Bank.

V15 preserves the scalar display Program and adds four explicit calculations: midpoint normalization, endpoint growth summaries, aligned-period statistics, and baseline z scores. These Programs can submit the computed final answer directly. They validate declared arithmetic, output scale and period alignment; they do not prove that the Agent chose the correct source column. Guidance covers column hierarchy, unit conversion and instrument scope separately.

Submission protocol `atomic.scalar-list-or-tool.v2` accepts complete bare flat lists as well as scalars and structured submissions. It preserves the whole answer text and rejects truncated responses, nested lists and surrounding prose. Legacy V1 behavior remains unchanged. Admission includes transport-specific separate reasoning costs in both the current request and the reserved finish.

Rebuild with `PYTHONPATH=src python resources/office_humanstrong_v15/tools/build_bundle.py`.

`experiments/run_office_humanstrong_v15.py` prepares one fixed 12-task diagnostic per existing model configuration after V14 settles. It uses a fresh immutable source and Bank snapshot, carries historical charges and unknown reservations, and does not run a new complete Test102 automatically. The 72 executions reserve at most 17,794,560 new tokens within the existing 270M cumulative ceiling. Qwen uses 400,000 tokens per task, including its separate 131,072 thinking allowance per request; the other five models use 216,576 per task. All use the existing Office 32,768 body/completion cap and 48 native calls.

The 12 selected tasks are exposed development failures and controls. Their diagnostic accuracy cannot establish the requested six-model full-Test threshold. The original scorer and old results remain unchanged, including inconsistencies in percent suffixes and trailing zeros.
