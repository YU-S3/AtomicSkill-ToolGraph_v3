"""Stop on Docker availability failures before model repair or answer scoring."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
import subprocess
from uuid import uuid4

from atomic_skillgraph.agents.provider import OpenAICompatibleProvider
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.checkpoint import save_json

VERSION = 'docker-availability-stop.v1'


class DockerUnavailable(RuntimeError):
    pass


def daemon_failure(result):
    diagnostic = result.get('diagnostic', '').lower()
    return result.get('status') == 'execution_error' and any(s in diagnostic for s in (
        'cannot connect to the docker daemon', 'error during connect:',
        'is the docker daemon running', 'permission denied while trying to connect to the docker'))


def docker_ready(image):
    try:
        result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerUnavailable(str(exc)) from exc
    if result.returncode or result.stdout.strip() != image:
        raise DockerUnavailable((result.stderr or result.stdout)[:4096])


@contextmanager
def infrastructure_guard(directory, image):
    original_execute, original_complete = ProgramWorker.execute, OpenAICompatibleProvider.complete
    original_run = Executor.run
    current_checkpoint = ContextVar('docker_guard_checkpoint', default=None)
    def stop(kind, details, checkpoint=None):
        path = Path(directory) / (uuid4().hex + '.json')
        save_json(path, {'version': VERSION, 'kind': kind, 'details': details,
                         'checkpoint_before': deepcopy(checkpoint.state) if checkpoint else None,
                         'policy': 'Stop without Agent repair or score; retain paid response/checkpoint.'})
        if checkpoint:
            checkpoint.advance(checkpoint.state['stage'], program_started=False,
                               infrastructure_reconstruction=str(path))
        raise DockerUnavailable('Docker unavailable; infrastructure receipt: ' + str(path))
    def run(executor, *args, **kwargs):
        token = current_checkpoint.set(getattr(executor, 'checkpoint', None))
        try:
            return original_run(executor, *args, **kwargs)
        finally:
            current_checkpoint.reset(token)
    def complete(provider, *args, **kwargs):
        try:
            docker_ready(image)
        except DockerUnavailable as exc:
            stop('before_model_request', str(exc))
        return original_complete(provider, *args, **kwargs)
    def execute(worker, program, inputs, broker, **kwargs):
        before = deepcopy(broker.events), broker.done
        result = original_execute(worker, program, inputs, broker, **kwargs)
        if daemon_failure(result):
            # Reconstruct only a pure Program that provably never reached the daemon.
            # Ambiguous interruptions retain the original UnknownSideEffect barrier.
            not_started = (result.get('execution_receipt', {}).get('exit_code') == 125
                and result.get('calls') == 0 and result.get('cpu_seconds') is None
                and 'cannot connect to the docker daemon' in result.get('diagnostic', '').lower()
                and not program.get('allowed_tools') and not getattr(broker.adapter, 'workspace', None)
                and before == (broker.events, broker.done))
            checkpoint = current_checkpoint.get() if not_started else None
            stop('worker_daemon_failure', {'worker_result': result, 'no_execution_proven': not_started,
                 'program_id': program['id']}, checkpoint)
        return result
    ProgramWorker.execute, OpenAICompatibleProvider.complete = execute, complete
    Executor.run = run
    try:
        yield
    finally:
        ProgramWorker.execute, OpenAICompatibleProvider.complete = original_execute, original_complete
        Executor.run = original_run
