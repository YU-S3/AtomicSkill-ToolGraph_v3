"""Pin local bytes for the unchanged upstream MiniLM embedding dependency."""
from pathlib import Path
import hashlib
import json
import subprocess

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def ensure_embedding_model(repo, python):
    root = repo / ".external/embedding_models/all-MiniLM-L6-v2"
    if not (root / "model.safetensors").is_file():
        # Distribution bootstrap downloads once. Worker execution is local-only.
        script = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'])"
        subprocess.run([str(python),"-c",script,MODEL_ID,REVISION,str(root)],check=True)
    files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts]
    required = {"modules.json","config.json","model.safetensors","tokenizer.json","1_Pooling/config.json"}
    hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    if not required.issubset(hashes):
        raise RuntimeError("Embedding snapshot is incomplete")
    lock = json.loads((repo / "experiments/baselines/b4_embodiskill/embedding_lock.json").read_text())
    if lock["model"] != MODEL_ID or lock["revision"] != REVISION or any(hashes.get(k) != v for k,v in lock["files"].items()):
        raise RuntimeError("Embedding bytes differ from pinned model snapshot")
    return dict(model=MODEL_ID, revision=REVISION, files=hashes, path=str(root))
