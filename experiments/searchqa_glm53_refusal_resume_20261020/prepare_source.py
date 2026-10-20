"""Version the provider fix without changing the running or historical releases."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
OLD = Path('/home/yangchengyu/asg_searchqa_glm53_v2_v7_release_20261020')
NEW = Path('/home/yangchengyu/asg_searchqa_glm53_refusal_release_20261020')
assert not NEW.exists()
subprocess.run(['git', 'clone', '--shared', str(OLD), str(NEW)], check=True)
subprocess.run(['git', 'checkout', '--detach', '53f264c4f185cdc24550b5c71844b36ea2d77c13'], cwd=NEW, check=True)
identity = subprocess.check_output(['git', 'log', '-1', '--format=%an%n%ae'], cwd=OLD, text=True).splitlines()
for field, value in zip(('user.name', 'user.email'), identity):
    subprocess.run(['git', 'config', field, value], cwd=NEW, check=True)
path = NEW / 'src/atomic_skillgraph/agents/provider.py'
text = path.read_text()
anchor = '        try:\n            usage, usage_metadata = _parse_usage(data.get("usage"))'
patch = '''        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish = choice.get("finish_reason")
        outcomes = {"content_filter": "provider_content_filter"}
        if self.config.dialect == "glm_53_chat":
            outcomes.update(sensitive="provider_content_filter", network_error="provider_inference_error",
                            model_context_window_exceeded="provider_context_window_exceeded")
        if finish in outcomes:
            # An explicit provider outcome remains identifiable even without billing.
            error = AgentProviderError(outcomes[finish], "Provider returned finish_reason=" + finish,
                                       http_status=response.status_code)
            try:
                billed, metadata = _parse_usage(data.get("usage"))
            except (TypeError, ValueError):
                pass
            else:
                error.usage_turn = AgentTurn(content="", tool_calls=[], latency_ms=latency_ms,
                    finish_reason=finish, prompt_tokens=billed["prompt_tokens"],
                    completion_tokens=billed["completion_tokens"], total_tokens=billed["total_tokens"],
                    reasoning_tokens=billed["reasoning_tokens"],
                    provider_metadata={**metadata, "response_model": data.get("model"),
                                       "response_id": data.get("id"), "provider_outcome": outcomes[finish]})
            raise error
'''
assert text.count(anchor) == 1
text = text.replace(anchor, patch + anchor)
start = text.index("        if self.config.dialect=='glm_53_chat' and finish_reason in")
end = text.index('        raw_calls = message.get("tool_calls", [])', start)
text = text[:start] + text[end:]
path.write_text(text)
subprocess.run(['git', 'add', str(path.relative_to(NEW))], cwd=NEW, check=True)
subprocess.run(['git', 'commit', '-m', 'Classify terminal provider outcomes before optional usage parsing'], cwd=NEW, check=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=NEW, text=True).strip()
diff = subprocess.check_output(['git', 'diff', 'HEAD^', 'HEAD'], cwd=NEW, text=True)
(HERE / 'source_diff.patch').write_text(diff)
(HERE / 'source_provenance.json').write_text(json.dumps({
    'source': str(NEW), 'parent': str(OLD), 'parent_commit': '53f264c4f185cdc24550b5c71844b36ea2d77c13',
    'release_commit': commit, 'provider_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'scope': 'Provider response classification only; unchanged payload, Bank, Runtime, scorer and model parameters.'
}, indent=2))
print(commit)
