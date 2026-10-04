"""Empirical SkillCompiler; independent of legacy proof and lifecycle policy."""

PROFILE = "skillcompiler.empirical.v1"

IMPLEMENTATION_REVISION = 'empirical-v3.1-CF2'
CF2_DEFAULTS = {
    'experiment': {'implementation_revision': IMPLEMENTATION_REVISION},
    'runtime': {'read_batch_max_calls': 3, 'result_inline_max_chars': 2048,
                'result_preview_max_items': 40, 'result_read_window_chars': 12000},
    'learning': {'builder_truncation_recovery_max_completion_tokens': 65536},
}
