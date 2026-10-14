"""Confirmed benchmark adaptation shared by formal and finite entry points."""
from copy import deepcopy


def benchmark_config(base, benchmark):
    config=deepcopy(base)
    if benchmark=='searchqa' and config['experiment'].get('searchqa_method_version'):
        from ..empirical.searchqa_guidance import VERSION, MATERIAL, POLICY
        if config['experiment']['searchqa_method_version']!=VERSION:raise ValueError('Unknown SearchQA method')
        config['learning'].update(searchqa_material_version=MATERIAL,searchqa_payload_max_bytes=65536)
        config['runtime']['searchqa_selection']=deepcopy(POLICY)
        config['program_worker']['record_execution_receipt']=True
    if benchmark=='officeqa':
        config['runtime'].update(finish_input_max_bytes=16384, dynamic_output_min_tokens=8192)
        config['llm'].setdefault('extractor',{})['protocol']={'thinking_type':'disabled'}
        config['llm']['extractor']['max_completion_tokens']=8192
        from ..empirical.source_catalog import enabled, trace_enabled
        from ..empirical import OFFICE_SOURCE_MATERIAL_VERSION, OFFICE_TRACE_POLICY
        if enabled(config):
            config.setdefault('learning',{}).update(material_version=OFFICE_SOURCE_MATERIAL_VERSION,
                source_program_payload_max_bytes=24576, builder_payload_max_bytes=24576)
            if trace_enabled(config):config['learning'].update(OFFICE_TRACE_POLICY)
            config.setdefault('budget',{}).update(eval_task_tokens=192000)
    if benchmark=='livemath':
        config['learning']['choice_guidance']={'enabled':True}
        config['runtime']['choice_guidance']={'enabled':True}
        if config['experiment'].get('choice_method_version'):
            config['learning']['choice_guidance']['public_process']=True
            config['runtime']['choice_guidance']['public_process']=True
        config['llm'].setdefault('purpose_overrides',{}).update({
            purpose:{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':cap}
            for purpose,cap in [('guidance_learning',2048),('guidance_grounding',1536)]})
        from ..empirical.math_capability import VERSIONS,configure
        if config['experiment'].get('choice_method_version') in VERSIONS:config=configure(config)
    return config
