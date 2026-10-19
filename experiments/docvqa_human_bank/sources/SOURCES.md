# Source and attribution notes

- Project/source ABI and split: https://github.com/YU-S3/AtomicSkill-ToolGraph_v3/tree/4cfa14296317a0900b2a832679b9e3fe0bb0f52b
  - `src/atomic_skillgraph/harness/benchmarks.py`
  - `src/atomic_skillgraph/agents/image_input.py`
  - `src/atomic_skillgraph/empirical/program_worker.py`
  - `src/atomic_skillgraph/empirical/contracts.py`
  - `src/atomic_skillgraph/harness/scorers/docvqa.py`
  - `configs/docvqa_deepseek_v41_base.yaml`, `docs/docvqa_deepseek_v41.md`
- Original strong public skill references, **not directly inserted as Ours learned output**:
  - https://github.com/microsoft/SkillOpt/blob/main/ckpt/docvqa/gpt5.5_skill.md (blob `476992fc1883b239b9dbca3d9498306a538c4bd9`)
  - https://github.com/EvolvingLMMs-Lab/SkillOpt-Lite/blob/main/skillopt_lite_ckpt/docvqa/gpt5.5/skill.md (blob `b4aa8ca9e0e6c49cb5f0faa702e90de037884ae5`)
  - https://github.com/microsoft/SkillOpt/blob/main/skillopt/envs/docvqa/rollout.py
- The concise reusable procedural descriptions are independently edited; no upstream evaluation answers, Test examples, hidden OCR text or proof sketches are included.
- Earlier user's initial experimental draft sometimes listed DocVQA Train180/Val24/Test200/Reserve130. Current production authoritative manifest is Train180/Val22/Test201/Reserve131, which this pack uses. The official original-image mode must be kept.
- Source code and container are original reference assets authored for this package. No pretrained OCR engine or pretrained model weights are redistributed.
