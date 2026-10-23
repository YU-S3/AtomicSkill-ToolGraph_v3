# OfficeQA human reference Bank V14

Version: `office.humanstrong.submission-facts.v14.20261023`.

This is a human-authored, development-exposed reference Bank in native AtomicSkill / Program / Implementation format. It is not naturally learned Ours or a blind Test result. The existing public-goal/source-text selector chooses at most six cards, including the answer contract; it does not invoke a learned Planner or inspect task IDs, gold answers or scores. Test-time learning is disabled.

The Bank retains the previous short rules and adds separate cards for final submission, arc change, symmetric percent difference, median absolute deviation, table boundaries, log-growth CV, complete monthly series, date encoding, category aggregation and public FFO-1/AY-1 structure. Specific cards replace conflicting generic cards. Facts describe public headers and row conventions, not task answers. Cards remain visible in both normal runtime and finalization.

`scalar_display` is the only Program. It reads an already acquired numeric leaf, performs explicit scale conversion and rounding, then submits directly. It never certifies that the caller selected the right data or formula. Rounding precision and fixed-width formatting are separate: redundant fractional zeros are removed by default; `format.fixed_decimals=true` retains them when explicitly required. The answer rule defaults to a number alone, with explicit templates/suffixes/list formats taking precedence.

The opt-in generic `atomic.scalar-or-tool.v1` protocol accepts an entire bare numeric `stop` response as a final submission without asking the model to rewrite it. Explanations, ambiguous alternatives and truncated responses are not extracted into answers. Finalization exposes one structured `finish_answer` interface. Existing configurations without this flag retain their previous behavior.

## Reproduce assets and run

```bash
PYTHONPATH=src python resources/office_humanstrong_v14/tools/build_bundle.py
PYTHONPATH=src python experiments/run_office_humanstrong_v14.py preview
PYTHONPATH=src python experiments/run_office_humanstrong_v14.py prepare --root NEW_RUN
PYTHONPATH=src python experiments/run_office_humanstrong_v14.py check --root NEW_RUN
PYTHONPATH=src python experiments/run_office_humanstrong_v14.py launch --root NEW_RUN
PYTHONPATH=src python experiments/run_office_humanstrong_v14.py status --root NEW_RUN
```

Paid preparation requires a clean committed source. One frozen canonical external-aligned Test102 is run for DeepSeek, Kimi and GLM, alternating models at task boundaries under one ledger. All use the same Bank, public pages and scoring. Per-task cap remains 216,576 tokens / 48 native calls; finalization uses thinking and a maximum 32,768 completion tokens. The user approved a cumulative 270M token ceiling on 2026-10-23. Historical costs and unknown reservations remain charged. No automatic repeats or score adjustment are authorized by this launcher.

Accuracy targets are at least 73/102 (71.57%), preferably 77/102 (75.49%) **for each model**, measured rather than assumed. Docker control failures stop the campaign; ordinary model errors retain their actual outcomes. All old runs and Bank versions remain unchanged.
