# Expression organization adaptation

The ordered grouping in `agents/skill_guidance.py::organize_guidance_view`
adapts the main/note traversal of EmbodiSkill's
`tasks/workflow/format.py::format_task_prompt_with_skills`.

Source: https://github.com/air-embodied-brain/EmbodiSkill
Pinned commit: `760126030eab1d33ec6a6f30988f0f1fb58df3a7`.
Source Git blob: `ac575cdc89c795580db7299cd96535854567ab5e`.
Copyright (c) 2026 EmbodiSkill contributors. MIT license:
`third_party/embodiskill/LICENSE`.

Adaptations use typed steps/notes and retain exact text, order, repetitions,
absence and unknown public fields. No upstream case folding, deduplication,
task examples, manual updates, reflection, retrieval or provider code is used.
The native-interface roundtrip logic is local R10.3 code, not an upstream
algorithm. No token or behavioral equivalence claim follows from formatting.
