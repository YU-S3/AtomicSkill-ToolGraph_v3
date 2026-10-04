# Third-party scoring code

The five files under `src/atomic_skillgraph/harness/scorers/` are copied without algorithm changes from Microsoft SkillOpt, pinned commit `fa4ca184573e42ec11472959dd57422381418096`:

- `searchqa.py` ← `skillopt/envs/searchqa/evaluator.py`
- `spreadsheet.py` ← `skillopt/envs/spreadsheetbench/evaluator.py`
- `officeqa.py` ← `skillopt/envs/officeqa/evaluator.py`
- `docvqa.py` ← `skillopt/envs/docvqa/evaluator.py`
- `livemath.py` ← `skillopt/envs/livemathematicianbench/evaluator.py`

Source: https://github.com/microsoft/SkillOpt/tree/fa4ca184573e42ec11472959dd57422381418096/skillopt/envs

MIT license and copyright are preserved in `src/atomic_skillgraph/harness/scorers/LICENSE`. File hashes are recorded in `benchmark_profiles.json`.

The local bridge separates public inputs from evaluator gold, seals spreadsheet code and outputs, recalculates formulas with the locked LibreOffice container, and performs all variant scoring without Agent feedback. These changes are execution adapters, not reproductions of the SkillOpt learning algorithm.

The previously adapted EmbodiSkill formatting implementation has been retired from the current production lane. Its code, attribution and license remain available in `archive/pre-empirical-v31`; see `docs/history/README.md`.
