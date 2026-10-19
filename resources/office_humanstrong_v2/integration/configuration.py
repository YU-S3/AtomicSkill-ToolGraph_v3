"""Independent evaluation config bound to the approved input snapshot."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from atomic_skillgraph.empirical.evidence_handoff import VERSION as HANDOFF
from atomic_skillgraph.empirical.system import validate_config

INPUT_VERSION='office.external-minimax.public-pages.v1'


def verify_resources(resources):
    root=Path(resources).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['version']!=INPUT_VERSION:
        raise ValueError('Approved external input snapshot required')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Resource snapshot changed: '+name)
    return root,manifest


def configure(base,resources,split):
    if split not in ('train','val','test'):raise ValueError('Unknown split')
    root,manifest=verify_resources(resources)
    cfg=deepcopy(base)
    cfg['manifest']=str(root/(split+'.json'))
    cfg['harness'].update(office_public_input_version=INPUT_VERSION,split=split,
        evaluator_records=str(root/('evaluator_records_'+split+'.json')))
    cfg['experiment'].update(office_public_input_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        office_human_bank_version='office.humanstrong.programmatic.v2.20261020',
        input_alignment='legacy_val_internal_only' if split=='val' else 'external_complete_public_prompt_and_evaluator')
    cfg['runtime'].update(evidence_handoff_version=HANDOFF,finish_input_max_bytes=65536,dynamic_output_min_tokens=8192,
                          compact_request_json=True)
    cfg['llm'].setdefault('purpose_overrides',{})['finish_only']={
        'protocol':{'thinking_type':'enabled'},'max_completion_tokens':8192}
    return validate_config(cfg)
