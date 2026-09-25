"""Execution-only concurrency: ordered results, isolated JVMs, shared HTTP cap."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from experiments.baselines.common.provider_gate import CampaignProviderGate


def workers():
    value=int(os.environ.get('SW_EPISODE_WORKERS','1'))
    if not 1<=value<=8: raise ValueError('SW_EPISODE_WORKERS must be 1..8')
    return value


def ordered_map(function, entries):
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        return list(pool.map(function, entries))


def provider_gate():
    root=os.environ.get('SW_GATE_ROOT')
    if not root:return None
    return CampaignProviderGate(gate_dir=Path(root)/'provider',campaign_id='scienceworld_parallel',
        max_inflight=int(os.environ['SW_PROVIDER_SLOTS']))


def environment_slot():
    root=os.environ.get('SW_GATE_ROOT')
    if not root:return nullcontext()
    gate=CampaignProviderGate(gate_dir=Path(root)/'environment',campaign_id='scienceworld_environments',
        max_inflight=int(os.environ['SW_ENV_SLOTS']))
    return gate.acquire(run_id=str(os.getpid()),seed=0,role='environment',stage='episode',logical_call_id=uuid.uuid4().hex)
