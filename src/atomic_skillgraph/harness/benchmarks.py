"""Public QA, corpus and spreadsheet interfaces; gold stays in the evaluator."""
import base64
import fnmatch
import importlib
import hashlib
import json
from pathlib import Path
import re
import tempfile

from ..empirical.contracts import digest, object_schema
from ..empirical.program_worker import ProgramWorker
from ..empirical.workspace import Workspace
from .simple_protocol import Broker, Capabilities
from .tool_spec import ToolSpec, result_schema


def truncate_context(context, max_chars=6000):
    if len(context) <= max_chars:
        return context
    result = ''
    for document in context.split('[DOC]'):
        candidate = result + '[DOC]' + document if result else document
        if len(candidate) > max_chars:
            break
        result = candidate
    return result or context[:max_chars] + '\n...[truncated]'


def score_answer(benchmark, prediction, record, *, audit=None):
    scorer = importlib.import_module('.scorers.' + benchmark, __package__)
    prediction = str(prediction or '')
    if benchmark == 'livemath':
        raw = scorer.evaluate(prediction, record['correct_choice'], record['choices'])
    elif benchmark == 'officeqa':
        matches = re.findall(r'<answer>(.*?)</answer>', prediction, flags=re.I | re.S)
        raw = scorer.evaluate(matches[-1].strip() if matches else prediction.strip(), record['answer'])
    else:
        raw = scorer.evaluate(prediction, record['answers'])
    if benchmark == 'docvqa':
        hard, soft, name = raw['anls'] >= .999, raw['anls'], 'docvqa.skillopt-anls'
    else:
        hard, soft, name = bool(raw['em']), raw['f1'], 'officeqa_skillopt_em_f1_v1' if benchmark == 'officeqa' else benchmark + '.skillopt-em-f1'
    if audit is not None:
        audit.update(raw_scorer_output=raw, scorer_version=hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest())
    return {'hard': hard, 'soft': soft, 'raw_score': soft, 'scorer': name}


class AnswerAdapter:
    capabilities = Capabilities(interaction='single_answer', tool_surface='none', final_submission_kind='single_answer')

    def __init__(self, benchmark, records):
        self.benchmark, self._records = benchmark, records
        if benchmark == 'docvqa':
            self.capabilities = Capabilities(interaction='single_answer', input_modalities=('text','image'), tool_surface='none', final_submission_kind='single_answer')

    def reset(self, task):
        self.task = task
        if task.task_id not in self._records:
            raise ValueError('Unknown evaluator task')
        return self.observe()

    def observe(self): return {'goal': self.task.goal, 'inputs': self.task.inputs}
    def available_tools(self): return []
    def tool_definitions(self): return []
    def close(self): pass

    def content_parts(self):
        parts = []
        for path in self.task.inputs.get('images', []):
            path = Path(path)
            mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
            parts.append({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' +
                base64.b64encode(path.read_bytes()).decode(), 'detail': 'high'}})
        return parts

    def call(self, name, arguments):
        raise ValueError('Single-answer tasks have no native tools')

    def submit(self, prediction): return {'task_id': self.task.task_id, 'prediction': prediction}

    def evaluate(self, sealed):
        if sealed['task_id'] != self.task.task_id:
            raise ValueError('Submission task mismatch')
        self.score_audit = {}
        return score_answer(self.benchmark, sealed['prediction'], self._records[self.task.task_id], audit=self.score_audit)


class FileAdapter(AnswerAdapter):
    capabilities = Capabilities(input_modalities=('text','files'), checkpoint_mode='workspace_copy', final_submission_kind='text')

    def tool_definitions(self): return [spec.view() for spec in self.specs.values()]
    def available_tools(self): return self.tool_definitions()

    def model_state(self):
        return {'inputs': self.task.inputs,
                'workspace': {k: v for k, v in self.observe()['workspace'].items() if k in {'outputs', 'content_hash'}}}

    def progress_key(self): return digest(self.model_state())

    def submission_ready(self, output, *, result_role='intermediate'):
        return self.capabilities.final_submission_kind == 'text' and isinstance(output, dict) and 'answer' in output and result_role == 'final_answer'

    def __init__(self, benchmark, records, config):
        super().__init__(benchmark, records)
        self.config = config
        self.worker = ProgramWorker(config.get('program_worker'))
        self._temporary = None

    def reset(self, task):
        super().reset(task)
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix='skillcompiler-workspace-')
        inputs = self._records[task.task_id].get('public_files', {})
        self.workspace = Workspace(self._temporary.name, inputs)
        return self.observe()

    def observe(self):
        manifest = self.workspace.root / 'manifest.json' if hasattr(self, 'workspace') else None
        return {'goal': self.task.goal, 'inputs': self.task.inputs,
                'workspace': json.loads(manifest.read_text()) if manifest and manifest.exists() else {}}

    def close(self):
        if self._temporary:
            # Published output versions are read-only; cleanup needs owned permissions.
            for path in Path(self._temporary.name).rglob('*'):
                if not path.is_symlink(): path.chmod(0o755 if path.is_dir() else 0o644)
            self._temporary.cleanup()
            self._temporary = None

    def declared_outputs(self, outputs): return outputs.get('files', [])

    def _recalculate(self, predicted):
        evaluator = FileAdapter('spreadsheet', {self.task.task_id: {'public_files': {'input.xlsx': predicted}}}, self.config)
        evaluator.reset(self.task)
        source = '''import subprocess, shutil
from pathlib import Path
result = subprocess.run(['libreoffice', '-env:UserInstallation=file:///tmp/lo-profile', '--headless',
    '--convert-to', 'xlsx', '--outdir', '/tmp/recalculated', INPUT_PATH], capture_output=True, timeout=90)
assert result.returncode == 0, result.stderr.decode(errors='replace')
shutil.copyfile('/tmp/recalculated/input.xlsx', OUTPUT_PATH)
'''
        result = evaluator._python(source, ['case1_result.xlsx'])
        if not result['accepted']:
            evaluator.close()
            raise RuntimeError('Formula recalculation failed: ' + str(result['error']))
        manifest = json.loads((evaluator.workspace.root/'manifest.json').read_text())
        return evaluator, evaluator.workspace.root/manifest['version']/'case1_result.xlsx'

    def _python(self, source, files, deleted_files=()):
        wrapper = 'def run(ctx, inputs):\n    namespace = {"INPUT_PATH": "/workspace/inputs/input.xlsx", "OUTPUT_PATH": "/workspace/case1_result.xlsx"}\n'
        wrapper += '    exec(' + repr(source) + ', namespace)\n'
        wrapper += '    return {"status": "ok", "outputs": {"files": inputs["files"], "deleted_files": inputs["deleted_files"]}}\n'
        output_fields = {'files': {'type': 'array', 'items': {'type': 'string'}}, 'deleted_files': {'type': 'array', 'items': {'type': 'string'}}}
        program = {'id': 'temporary_' + digest(wrapper), 'source': wrapper, 'entry': 'run',
            'input_schema': object_schema(output_fields, ['files', 'deleted_files']),
            'output_schema': object_schema(output_fields, ['files', 'deleted_files']),
            'allowed_tools': [], 'environment': self.config['program_environment']}
        result = self.worker.execute(program, {'files': files, 'deleted_files': list(deleted_files)}, Broker(self, 1))
        return {'accepted': result['status'] == 'ok', 'observation': result.get('diagnostic', ''),
                'data': result.get('outputs', {}), 'error': result.get('detail'), 'done': False}


class OfficeAdapter(FileAdapter):
    def __init__(self, records, config):
        super().__init__('officeqa', records, config)
        self.corpus = Path(config['harness']['corpus_root']).resolve(strict=True)
        self.specs = {name: ToolSpec(name, description, schema, result_schema(data), effect, batchable, units)
            for name, description, schema, data, effect, batchable, units in [
            ('glob', 'List matching relative corpus paths', object_schema({'pattern': {'type':'string'}}, ['pattern']), {'type': 'array', 'items': {'type': 'string'}}, 'read_only', True, {}),
            ('read', 'Read at most 12000 characters. offset is a zero-based decoded Unicode character index, not a line or byte position.',
             object_schema({'path': {'type':'string'}, 'offset': {'type':'integer','minimum':0}}, ['path','offset']), {'type': 'object'}, 'read_only', True, {'offset': 'unicode_codepoint_0_based'}),
            ('grep', 'Search corpus; up to 40 hits. line is one-based display only; offset can be passed directly to read.',
             object_schema({'pattern': {'type':'string'}}, ['pattern']), {'type': 'array', 'items': {'type': 'object'}}, 'read_only', True, {'line': 'line_1_based', 'offset': 'unicode_codepoint_0_based'}),
            ('execute_python', 'Compute with acquired public values; no corpus or network mount.', object_schema({'source': {'type':'string'}}, ['source']), {'type': 'object'}, 'sandbox_compute', False, {})]}

    def _path(self, relative):
        path = (self.corpus / relative).resolve()
        if not path.is_relative_to(self.corpus):
            raise ValueError('Corpus path is not authorized')
        if not path.exists() or not path.is_file(): raise ValueError('Corpus file is missing or not a file')
        return path

    def call(self, name, arguments):
        if not self.corpus.is_dir(): raise RuntimeError('Authorized corpus root is unavailable')
        try:
            return self._call(name, arguments)
        except UnicodeError:
            raise
        except (ValueError, re.error) as exc:
            return {'accepted': False, 'observation': '', 'data': [] if name in {'glob','grep'} else {}, 'error': str(exc),
                    'error_code': 'invalid_corpus_input', 'done': False}

    def _call(self, name, arguments):
        if name == 'execute_python': return self._python(arguments['source'], [])
        if name == 'glob':
            data = sorted(p.relative_to(self.corpus).as_posix() for p in self.corpus.rglob('*.txt')
                          if fnmatch.fnmatch(p.relative_to(self.corpus).as_posix(), arguments['pattern']))
        elif name == 'read':
            text = self._path(arguments['path']).read_text(encoding='utf-8')
            offset = arguments['offset']
            data = {'text': text[offset:offset+12000], 'next_offset': offset+12000 if len(text)>offset+12000 else None}
        elif name == 'grep':
            pattern, data = re.compile(arguments['pattern'], re.I), []
            for path in sorted(self.corpus.rglob('*.txt')):
                path = self._path(path.relative_to(self.corpus).as_posix())
                offset = 0
                for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(keepends=True)):
                    if pattern.search(line):
                        data.append({'path': path.relative_to(self.corpus).as_posix(), 'line': index+1, 'offset': offset, 'text': line.rstrip('\n')[:1000]})
                        if len(data) == 40: break
                    offset += len(line)
                if len(data) == 40: break
        else:
            raise ValueError('Unknown corpus tool')
        return {'accepted': True, 'observation': '', 'data': data, 'error': None, 'done': False}


class SpreadsheetAdapter(FileAdapter):
    capabilities = Capabilities(input_modalities=('text','files'), checkpoint_mode='workspace_copy', final_submission_kind='files')

    def __init__(self, records, config):
        super().__init__('spreadsheet', records, config)
        spec = ToolSpec('execute_python', 'Run Python in the shared sandbox. INPUT_PATH and OUTPUT_PATH are predefined. '
            'Write solution.py using INPUT_PATH/OUTPUT_PATH and case1_result.xlsx. Declare output filenames to commit them.',
            object_schema({'source': {'type':'string'}, 'files': {'type':'array','items':{'type':'string'}},
                           'deleted_files': {'type':'array','items':{'type':'string'}}}, ['source','files']),
            result_schema({'type': 'object'}), effect='sandbox_compute')
        self.specs = {spec.name: spec}

    def submission_ready(self, output=None, *, result_role='intermediate', previous_workspace=None):
        pointer = self.workspace.root / 'manifest.json'
        if not pointer.exists(): return False
        manifest = json.loads(pointer.read_text())
        required = {'solution.py', 'case1_result.xlsx'}
        ready = all(name in manifest['outputs'] and (self.workspace.root/manifest['version']/name).is_file() for name in required)
        if previous_workspace is not None:
            ready = ready and required.issubset((output or {}).get('files', [])) and any(
                previous_workspace.get('hashes', {}).get(name) != manifest.get('hashes', {}).get(name) for name in required)
        return ready

    def call(self, name, arguments):
        if name != 'execute_python': raise ValueError('Unknown spreadsheet tool')
        return self._python(arguments['source'], arguments['files'], arguments.get('deleted_files', []))

    def submit(self, prediction):
        pointer = self.workspace.root / 'manifest.json'
        if not pointer.exists(): return {'task_id': self.task.task_id, 'bundle': None}
        manifest = json.loads(pointer.read_text())
        version = self.workspace.root / manifest['version']
        needed = ['solution.py', 'case1_result.xlsx']
        if not all(name in manifest['outputs'] and (version/name).is_file() for name in needed):
            return {'task_id': self.task.task_id, 'bundle': None}
        solution = {'entry': 'INPUT_PATH/OUTPUT_PATH', 'code_sha256': hashlib.sha256((version/'solution.py').read_bytes()).hexdigest(),
                    'image_digest': self.config['program_environment']['image_digest'],
                    'public_input_contract': {'input': 'input.xlsx', 'output': 'case1_result.xlsx'}}
        # Host seals metadata; the evaluator never sends results back to Agent.
        bundle = Path(tempfile.mkdtemp(prefix='skillcompiler-solution-'))
        import shutil
        for name in needed: shutil.copyfile(version/name, bundle/name)
        (bundle/'solution_manifest.json').write_text(json.dumps(solution))
        for path in bundle.iterdir(): path.chmod(0o444)
        bundle.chmod(0o555)
        return {'task_id': self.task.task_id, 'bundle': str(bundle)}

    def evaluate(self, sealed):
        from .scorers.spreadsheet import evaluate
        if sealed['task_id'] != self.task.task_id: raise ValueError('Submission task mismatch')
        record = self._records[self.task.task_id]
        cases = record['cases']
        outcomes = []
        raw_outputs = []
        def record_score(calculated, case):
            raw = evaluate(str(calculated), case['gold'], record['instruction_type'], record['answer_position'])
            raw_outputs.append(raw)
            return bool(raw['ok'])
        bundle = Path(sealed['bundle']) if sealed.get('bundle') else None
        if bundle:
            lock = json.loads((bundle/'solution_manifest.json').read_text())
            if lock['code_sha256'] != hashlib.sha256((bundle/'solution.py').read_bytes()).hexdigest() or lock['image_digest'] != self.config['program_environment']['image_digest']:
                raise ValueError('Sealed solution code/image changed')
        for index, case in enumerate(cases):
            predicted = bundle/'case1_result.xlsx' if bundle else None
            if bundle and index:
                evaluator = SpreadsheetAdapter({self.task.task_id: {**record, 'public_files': {'input.xlsx': case['input']}}}, self.config)
                try:
                    evaluator.reset(self.task)
                    source = (bundle/'solution.py').read_text()
                    # The upstream executor also replaces top-level path assignments.
                    source = re.sub(r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$', '', source, flags=re.M)
                    result = evaluator._python(source, ['case1_result.xlsx'])
                    if result['accepted']:
                        manifest = json.loads((evaluator.workspace.root/'manifest.json').read_text())
                        predicted = evaluator.workspace.root/manifest['version']/'case1_result.xlsx'
                        recalculator, calculated = self._recalculate(predicted)
                        try:
                            outcomes.append(record_score(calculated, case))
                        finally: recalculator.close()
                    else:
                        outcomes.append(False)
                        raw_outputs.append({'not_scored': 'variant_execution_failed', 'execution_result': result})
                finally: evaluator.close()
            elif predicted and predicted.exists():
                recalculator, calculated = self._recalculate(predicted)
                try:
                    outcomes.append(record_score(calculated, case))
                finally: recalculator.close()
            else:
                outcomes.append(False)
                raw_outputs.append({'not_scored': 'missing_prediction'})
        soft = sum(outcomes)/len(cases) if cases else 0
        self.score_audit = {'raw_scorer_output': raw_outputs, 'scorer_version': hashlib.sha256(
            Path(importlib.import_module('.scorers.spreadsheet', __package__).__file__).read_bytes()).hexdigest()}
        return {'hard': bool(cases) and all(outcomes), 'soft': soft, 'raw_score': soft,
                'scorer': 'spreadsheet.skillopt-all-cases', 'case_results': outcomes}


def create_adapter(config):
    name = config['harness']['adapter']
    records = json.loads(Path(config['harness']['evaluator_records']).read_text())
    if name == 'officeqa': return OfficeAdapter(records, config)
    if name == 'spreadsheet': return SpreadsheetAdapter(records, config)
    if name in {'searchqa','docvqa','livemath'}: return AnswerAdapter(name, records)
    raise ValueError('Unknown simple adapter: ' + name)
