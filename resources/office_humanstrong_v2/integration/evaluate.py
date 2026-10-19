"""Bind the frozen human Bank to the native evaluator and existing campaign ledger."""
import importlib.util
import hashlib
import json
from pathlib import Path

from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.experiments.run_empirical import run

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('office_v2_evaluation_frontier',ROOT/'frontier.py')
frontier=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontier)
spec=importlib.util.spec_from_file_location('office_v2_evaluation_config',ROOT/'configuration.py')
configuration=importlib.util.module_from_spec(spec);spec.loader.exec_module(configuration)


def evaluate(config,tasks,output,*,budget_governor,arm='manual',formal_log=None,resume=False):
    if arm not in ('manual','off'):raise ValueError('Choose manual or off')
    if budget_governor is None:raise ValueError('An existing campaign budget governor is required')
    if config['harness'].get('office_public_input_version')!=frontier.input_protocol.VERSION:
        raise ValueError('External aligned config required')
    resource,_=configuration.verify_resources(Path(config['manifest']).parent)
    if hashlib.sha256((resource/'manifest.json').read_bytes()).hexdigest()!=config['experiment']['office_public_input_manifest_sha256']:
        raise ValueError('External input manifest identity changed')
    if not all(isinstance(t,PublicTask) and t.inputs.get('office_public_input_version')==frontier.input_protocol.VERSION for t in tasks):
        raise ValueError('External aligned public tasks required')
    frozen={t['task_id']:PublicTask(**t) for t in json.loads(Path(config['manifest']).read_text())['tasks']}
    if any(frozen.get(t.task_id)!=t for t in tasks):raise ValueError('Task differs from its frozen split')
    bundle=json.loads((Path(config['data_dir'])/'imported_bank.json').read_text())
    if bundle['method_version']!=frontier.VERSION:raise ValueError('Wrong frozen human Bank')
    with frontier.bridge():
        return run(config,tasks,output,readonly=True,resume=resume,formal_log=formal_log,
            bank_view_factory=lambda bank:frontier.HumanBankView(bank,bundle,arm),
            budget_governor=budget_governor,freeze_on_complete=False)
