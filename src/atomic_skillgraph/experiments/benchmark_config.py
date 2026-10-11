"""Confirmed benchmark adaptation shared by formal and finite entry points."""
from copy import deepcopy


def benchmark_config(base, benchmark):
    config=deepcopy(base)
    if benchmark=='officeqa':
        config['runtime'].update(finish_input_max_bytes=16384, dynamic_output_min_tokens=8192)
        config['llm'].setdefault('extractor',{})['protocol']={'thinking_type':'disabled'}
        config['llm']['extractor']['max_completion_tokens']=8192
    if benchmark=='livemath':
        config['learning']['choice_guidance']={'enabled':True}
        config['runtime']['choice_guidance']={'enabled':True}
        config['llm'].setdefault('purpose_overrides',{}).update({
            purpose:{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':cap}
            for purpose,cap in [('guidance_learning',2048),('guidance_grounding',1536)]})
    return config
