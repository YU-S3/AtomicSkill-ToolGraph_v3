"""Export inspectable authored drafts, never a deployable Frozen snapshot."""
import argparse
from pathlib import Path

from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.deployment.scienceworld_reference import reference_atomics
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.tooling.validator import ToolStaticValidator


def export(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    harness = ScienceWorldAdapter()
    validator = ToolStaticValidator()
    inventory, validation = [], []
    assets = reference_atomics()
    for number,(atomic,implementation,tool) in enumerate(assets,1):
        check = validator.validate_tool_asset(tool,atomic,harness)
        if not check.passed:
            raise ValueError((str(tool.ref),check.failure_codes,check.messages))
        inventory.append({'id':f'A{number:02d}','atomic_ref':str(atomic.ref),
            'implementation_ref':str(implementation.ref),'tool_ref':str(tool.ref),
            'status':'draft','real_publication_qualified':False})
        validation.append({'tool_ref':str(tool.ref),**to_primitive(check)})
    output.mkdir(parents=True,exist_ok=False)
    for number,assets_for_atomic in enumerate(assets,1):
        for kind,asset in zip(('atomic','implementation','tool'),assets_for_atomic):
            assert asset.status.value == 'draft'
            atomic_write_json(output/'bank'/kind/f'A{number:02d}.json',to_primitive(asset))
    atomic_write_json(output/'audit'/'reference_inventory.json',inventory)
    atomic_write_json(output/'audit'/'static_validation.json',validation)
    atomic_write_json(output/'audit'/'release_checks.json',{
        'go':False,'experiment_kind':'authored_reference','deployable_frozen_bank':False,
        'atomic_implementation_tool_triples':len(assets),'composites':0,
        'blockers':['G01-G18 official task-contract/publication boundary awaits clarification',
                    'Independent real executions and SW-WF01 qualification not complete'],
        'limitations':['A20 count=0 supplies no new time.progressed witness and must not fabricate one'],
        'learned_train_asset_source':False})
    return inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    export(parser.parse_args().output)
