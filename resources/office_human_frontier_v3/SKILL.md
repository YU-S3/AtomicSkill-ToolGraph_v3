# Authored Office Bank V3

Runtime reads bank.json guidance; this file mirrors it. At most two phase cards are visible. Frozen manual assets, not learned usable claims.

## OA-Target

Lock the exact measure, observation periods, fiscal/calendar basis, reporting vintage, requested statistic, output unit and format. Source amounts only from acquired evidence. Search hits locate a source; they are not automatically the correct table, column or value. Focus on the single missing target field.

## OA-Locate

Use a narrow file scope and distinctive measure. The scoped_search Program can batch search and read real hit paths/offsets within this task budget. Inspect its acquired windows even when status=needs_input: that status means table/period alignment remains. After empty hits vary one query dimension. Read the exact hit and nearby header/unit; do not repeatedly expand to the whole corpus or treat an exhausted search as complete target coverage.

## OA-Align

Align the title, header, row, footnotes, period and unit before using adjacent OCR numbers. Use table_extract only with a complete relevant table window and exact column/row labels. Ambiguous OCR needs native read_context, not guessed cells. Keep each operand with its source_ref and exact quote in evidence_state; name it by its computation field (e.g. numerator, denominator, from_value, to_value, values.0). Preserve period order and missing observations; never fill gaps. Gross/net, offered/accepted, old/new securities, currency quotation direction and 32nds are different quantities, not interchangeable units.

## OA-Compute

Use series_metrics for its listed formulas when it fits; otherwise use existing temporary Python. Difference names FROM and TO plus absolute; ratio names numerator and denominator; change/growth names base_value and new_value. Set output_percent explicitly, never multiply twice. periods means elapsed intervals, not number of observations. Programs prove local arithmetic only. Persist the computed output via answer_candidate.result_ref with matching operation, output_unit and covers_task. Temporary Python may return operand_values alongside answer to retain its binding. If the target is solved, finish now rather than search again for the same quantity.

## OA-Commit

If the exact requested target has been computed from the required observations, immediately use native runtime_step(action="finish", answer=<only the requested numeric/list/date payload>). No XML tags. Do not restart a broad search or replace a grounded value with an unverified one. A candidate is not semantic certification: check its target, direction, period, unit and precision against its actual evidence. When forced to finish, preserve the best genuinely supported result and remaining uncertainty; do not invent a completion receipt.
