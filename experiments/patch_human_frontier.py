"""Apply the narrow protocol fix to a separate copy of the supplied manual Bank package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_SHA256 = '928c72fe227bed84de190f48d1385058fd80671337c8a69fa53d6c2bcf83990b'
VERSION = 'human.frontier.runtime.20261015.v1.finish-r1'


def patch(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'FILE_SHA256.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original package hash mismatch: '+name)
    code = source/'integration/frontier.py'
    if hashlib.sha256(code.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Patch requires the reviewed original bridge')
    text = code.read_text()
    anchor = "        task=getattr(s.adapter,'task',None)"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "        # Finish evidence was already fitted against the exact native wire payload.\n"
        "        if kwargs.get('decision_purpose')=='finish_only' or name=='finish_answer':\n"
        "            return original_agent(s,stage,prompt,material,name,schema,**kwargs)\n"+anchor)
    text = text.replace("                policy=s.bank.policy(task)",
        "                native_prompt=prompt\n                prompt=''\n                policy=s.bank.policy(task)")
    anchor = "                material=deepcopy(material)"
    assert text.count(anchor) == 1
    text = text.replace(anchor,
        "                prompt+='\\n\\nAuthored guidance governs task strategy; the current request governs the response protocol. Follow its tool schema and submission format, not authored answer-tag suggestions.\\n\\n'+native_prompt\n"+anchor)
    text = text.replace("VERSION = 'human.frontier.runtime.20261015.v1'", "VERSION = '"+VERSION+"'")
    compile(text, 'integration/frontier.py', 'exec')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination/'integration/frontier.py').write_text(text)
    (destination/'FILE_SHA256.original.json').write_bytes((source/'FILE_SHA256.json').read_bytes())
    updated = {**manifest, 'integration/frontier.py': hashlib.sha256(text.encode()).hexdigest()}
    (destination/'FILE_SHA256.json').write_text(json.dumps(updated, indent=2)+'\n')
    receipt = {'version':VERSION, 'original_bridge_sha256':SOURCE_SHA256,
        'patched_bridge_sha256':updated['integration/frontier.py'], 'changed_package_files':['integration/frontier.py'],
        'bank_sha256':manifest['bank/bank.json'], 'original_package':str(source.resolve())}
    (destination/'PATCH_PROVENANCE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(patch(args.source,args.destination)))
