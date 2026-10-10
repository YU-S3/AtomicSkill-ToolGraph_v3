"""Empirical SkillCompiler; independent of legacy proof and lifecycle policy."""

PROFILE = "skillcompiler.empirical.v1"

IMPLEMENTATION_REVISION = 'empirical-v3.2-atomic-unified'
LOCAL_VALIDATION_POLICY_VERSION = 'atomic.local-validation.v2'
ANSWER_PROTOCOL_VERSION = 'atomic.answer-executor.v1'
LEARNING_MATERIAL_VERSION = 'learner.material.atomic-unified.v1'
GUIDANCE_POLICY_VERSION = 'guidance.public-evidence.v1'
CHOICE_GUIDANCE_POLICY_VERSION = 'choice-guidance.grounded.v1'
CHOICE_GUIDANCE_MATERIAL_VERSION = 'choice-guidance.material.v1'
CHOICE_GUIDANCE_SELECTION_VERSION = 'choice-guidance.scope.v1'
SINGLE_ANSWER_PROMPT_VERSION = 'single-answer.semantic.v2'
CHOICE_NORMALIZER_VERSION = 'choice-guidance.normalizer.v1'
POLICY_DEFAULTS = {
    'experiment': {'implementation_revision': IMPLEMENTATION_REVISION},
    'runtime': {'read_batch_max_calls': 3, 'result_inline_max_chars': 2048,
                'model_view_version': 'empirical.model-view.atomic-unified.v1',
                'result_preview_max_items': 40, 'result_read_window_chars': 12000,
                'plan_execution_policy': 'empirical.recoverable-takeover.v1', 'dynamic_escape_limit': 1,
                'answer_protocol_version':ANSWER_PROTOCOL_VERSION},
    'learning': {'builder_truncation_recovery_max_completion_tokens': 32768,
                 'material_version': LEARNING_MATERIAL_VERSION,
                 'guidance_policy_version': GUIDANCE_POLICY_VERSION,
                 'min_distinct_train_cases_before_first_build': 1,
                 'local_validation_policy':LOCAL_VALIDATION_POLICY_VERSION},
}
