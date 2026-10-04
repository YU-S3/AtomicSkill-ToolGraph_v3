"""Digest-pinned container worker and bounded RPC to the episode owner."""
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4


def _worker():
    output, input_stream = sys.stdout, sys.stdin
    limit, invocation_id = int(sys.argv[2]), sys.argv[3]
    sequence = 0
    def send(value):
        encoded = json.dumps(value, allow_nan=False).encode() + b'\n'
        if len(encoded) > limit:
            raise ValueError('RPC message limit exceeded')
        output.buffer.write(encoded)
        output.flush()
    def rpc(method, **values):
        nonlocal sequence
        sequence += 1
        send({'kind': 'rpc', 'invocation_id': invocation_id, 'sequence': sequence,
              'method': method, 'values': values})
        raw = input_stream.buffer.readline(limit + 1)
        if len(raw) > limit or not raw.endswith(b'\n'):
            raise ValueError('Invalid broker response')
        response = json.loads(raw)
        if 'rpc_error' in response:
            raise RuntimeError(response['rpc_error'])
        return response['value']
    class Context:
        def observe(self): return rpc('observe')
        def available_tools(self): return rpc('available_tools')
        def remaining_calls(self): return rpc('remaining_calls')
        def call(self, name, arguments): return rpc('call', name=name, arguments=arguments)
        def read_result(self, result_id, offset=0, limit=None, path=None):
            values = {'result_id': result_id, 'offset': offset}
            if limit is not None: values['limit'] = limit
            if path is not None: values['path'] = path
            return rpc('read_result', **values)
    source = Path('/program/source.py').read_text()
    inputs = json.loads(Path('/program/inputs.json').read_text())
    sys.stdout = sys.stderr
    try:
        namespace = {'__name__': 'generated_program'}
        exec(compile(source, '/program/source.py', 'exec'), namespace)
        result = namespace['run'](Context(), inputs)
        if not isinstance(result, dict) or result.get('status') not in {'ok', 'not_found', 'needs_input', 'blocked'}:
            raise ValueError('run must return a valid status dictionary')
        if not isinstance(result.get('outputs', {}), dict):
            raise ValueError('outputs must be a dictionary')
        send({'kind': 'done', 'result': result})
    except BaseException as exc:
        send({'kind': 'done', 'result': {'status': 'execution_error', 'detail': str(exc)[:2048]}})


class ProgramWorker:
    def __init__(self, settings=None):
        self.settings = {'max_tool_calls_per_invocation': 32, 'wall_timeout_seconds': 60,
                         'memory_limit_mb': 1024, 'max_rpc_message_bytes': 1048576,
                         'pids_limit': 64, 'cpus': 1, **(settings or {})}
        self.invocations = []

    def command(self, image, root, container, workspace=None):
        s = self.settings
        args = ['docker', 'run', '--rm', '-i', '--pull=never', '--name', container,
                '--network=none', '--read-only', '--user=1000:1000', '--cap-drop=ALL',
                '--security-opt=no-new-privileges', '--memory', str(s['memory_limit_mb']) + 'm',
                '--memory-swap', str(s['memory_limit_mb']) + 'm', '--pids-limit', str(s['pids_limit']),
                '--cpus', str(s['cpus']), '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m',
                '--mount', 'type=bind,src=' + str(root) + ',dst=/program,readonly']
        if workspace:
            args += ['--mount', 'type=bind,src=' + str(workspace) + ',dst=/workspace',
                     '--mount', 'type=bind,src=' + str(workspace / 'inputs') + ',dst=/workspace/inputs,readonly']
        return args + ['--workdir', '/workspace' if workspace else '/tmp', image,
                       'python', '-I', '-u', '/program/worker.py', '--worker',
                       str(s['max_rpc_message_bytes']), container]

    def execute(self, program, inputs, broker):
        from .contracts import validate_schema_instance
        from ..harness.simple_protocol import UnknownSideEffect
        validate_schema_instance(inputs, program['input_schema'])
        image = program.get('environment', {}).get('image_digest')
        if not isinstance(image, str) or not (image.startswith('sha256:') or '@sha256:' in image):
            raise RuntimeError('Program requires a locked container image digest')
        if sys.platform != 'linux' or not shutil.which('docker'):
            raise RuntimeError('sandbox_python_v1 requires Docker on Linux; no host execution fallback')
        s = self.settings
        if len(json.dumps(inputs, ensure_ascii=False, allow_nan=False).encode()) > s['max_rpc_message_bytes']:
            raise ValueError('Program inputs exceed RPC limit; use bounded read_result with a result_id input')
        container = 'skillcompiler-' + uuid4().hex
        calls, result, diagnostic = 0, {'status': 'execution_error', 'detail': 'Worker exited without a result'}, bytearray()
        started = time.monotonic()
        deadline = started + s['wall_timeout_seconds']
        workspace = getattr(broker.adapter, 'workspace', None)
        stage = workspace.stage() if workspace else None
        broker.open_lease(container)
        try:
            with tempfile.TemporaryDirectory(prefix='skillcompiler-program-') as directory:
                root = Path(directory)
                root.chmod(0o755)
                (root / 'source.py').write_text(program['source'], encoding='utf-8')
                (root / 'inputs.json').write_text(json.dumps(inputs, allow_nan=False), encoding='utf-8')
                shutil.copyfile(__file__, root / 'worker.py')
                for path in root.iterdir(): path.chmod(0o444)
                process = subprocess.Popen(self.command(image, root, container, stage), stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ, 'rpc')
                selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
                buffer, finished, seen = b'', False, set()
                try:
                    while not finished and selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            result = {'status': 'execution_error', 'detail': 'Worker wall timeout'}
                            break
                        for key, _ in selector.select(min(remaining, .2)):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if key.data == 'stderr':
                                diagnostic.extend(chunk[:max(0, 4096 - len(diagnostic))])
                                continue
                            buffer += chunk
                            if len(buffer) > s['max_rpc_message_bytes']:
                                raise ValueError('Worker RPC message limit exceeded')
                            while b'\n' in buffer:
                                line, buffer = buffer.split(b'\n', 1)
                                request = json.loads(line)
                                if request.get('kind') == 'done':
                                    result, finished = request['result'], True
                                    break
                                if request.get('invocation_id') != container or not isinstance(request.get('sequence'), int):
                                    raise ValueError('Invalid invocation RPC identity')
                                sequence, method = request['sequence'], request.get('method')
                                if method in {'call', 'read_result'} and sequence not in seen:
                                    if calls >= s['max_tool_calls_per_invocation']:
                                        raise RuntimeError('Program native call budget exhausted')
                                    calls += 1
                                seen.add(sequence)
                                try:
                                    value = broker.rpc(container, sequence, method, request.get('values', {}),
                                        deadline=deadline, allowed_tools=program['allowed_tools'])
                                    if method == 'remaining_calls': value = min(value, s['max_tool_calls_per_invocation'] - calls)
                                    response = {'value': value}
                                except (ValueError, PermissionError) as exc:
                                    response = {'rpc_error': str(exc)}
                                encoded = json.dumps(response, allow_nan=False).encode() + b'\n'
                                if len(encoded) > s['max_rpc_message_bytes']:
                                    raise ValueError('Broker result must use a resource handle')
                                process.stdin.write(encoded)
                                process.stdin.flush()
                except UnknownSideEffect:
                    raise
                except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
                    result = {'status': 'execution_error', 'detail': str(exc)}
                finally:
                    broker.close_lease(container)
                    selector.close()
                    subprocess.run(['docker', 'rm', '-f', container], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10)
                    if process.poll() is None: process.kill()
                    process.wait(timeout=10)
                    for stream in (process.stdin, process.stdout, process.stderr): stream.close()
            if not isinstance(result, dict) or result.get('status') not in {'ok', 'not_found', 'needs_input', 'blocked', 'execution_error'}:
                raise ValueError('Invalid worker result')
            if result['status'] == 'ok':
                validate_schema_instance(result.get('outputs'), program['output_schema'])
                if workspace:
                    result['workspace'] = workspace.publish(stage, broker.adapter.declared_outputs(result['outputs']),
                                                            result['outputs'].get('deleted_files', []))
                    stage = None
            else:
                result.pop('outputs', None)
        except UnknownSideEffect:
            raise
        except (ValueError, OSError, TypeError, KeyError) as exc:
            result = {'status': 'execution_error', 'detail': str(exc)}
        finally:
            broker.close_lease(container)
            if stage is not None: workspace.discard(stage)
        result = {**result, 'calls': calls, 'elapsed_seconds': time.monotonic() - started,
                  'diagnostic': diagnostic.decode(errors='replace'), 'program_id': program['id']}
        self.invocations.append(result)
        return result


if __name__ == '__main__' and sys.argv[1:2] == ['--worker']:
    _worker()
