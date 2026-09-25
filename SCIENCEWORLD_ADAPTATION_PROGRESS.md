# ScienceWorld baseline adaptation checkpoint

Updated 2026-09-25. Baseline source starts at 45d1bdb5f9fbbf2d11279e1be4c1ca6be8bcd684.
Local branch baselines-skillopt publishes to the existing remote baseline-skillopt
branch of YU-S3/AtomicSkill-ToolGraph_v3. No pull request is required.

Implemented independent ScienceWorld 1.2.3 public-frame adapters for B0 Dynamic,
B1 StaticSkill, B3 SkillOpt, B4 EmbodiSkill, and B5 GEPA. The original pinned
learning algorithms are retained. The official Train120/Dev10/Test90 manifests
match main byte-for-byte. Test is not exposed to optimizers. Scores preserve the
official continuous reward, including negative failures. Provider usage includes
retained failed attempts; canonical action paths name a single completed attempt.

Environment: /home/yangchengyu/asg_scienceworld_venv with OpenJDK 17; the original
ALFWorld environment is unchanged. Optional dependency: pip install -e '.[scienceworld]'.
Original pinned .external dependencies and the EmbodiSkill dependency environment
are still required; they are not replaced by new learning implementations.

Validation: 895 passed, 4 skipped in the complete baseline suite. The five real-API
diagnostic lanes completed under
/home/yangchengyu/asg_scienceworld_baseline_acceptance_20260925/.
B0/B1 evaluated three shared diagnostic test cases; B3/B4/B5 additionally exercised
two Train and two Dev cases before the three frozen test cases. These are wiring
diagnostics, not formal benchmark results or evidence of full training coverage.
Detailed final suite log: /home/yangchengyu/scienceworld_baseline_final_pytest.log.

Main's public Tool IR extensions belong to its Execution Graph/Tool interpreter;
these text/skill baseline algorithms do not execute those programs and must not
be silently converted into our method. Baselines independently contain the shared
public benchmark adapter, not a runtime import from the main working directory.

No formal experiment has been launched. Full six-line formal release still needs
main's complete reference Bank, Train-only Bank Compiler and final release gates.
