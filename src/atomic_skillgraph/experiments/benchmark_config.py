"""Confirmed benchmark adaptation shared by formal and finite entry points."""
from copy import deepcopy


def benchmark_config(base, benchmark):
    config=deepcopy(base)
    if benchmark=='livemath':
        config['learning']['choice_guidance']={'enabled':True}
        config['runtime']['choice_guidance']={'enabled':True}
        config['llm'].setdefault('purpose_overrides',{}).update({
            purpose:{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':cap}
            for purpose,cap in [('guidance_learning',2048),('guidance_grounding',1536)]})
    return config
