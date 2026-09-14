"""Hold real GEPA/SkillOpt/ALFWorld workers during the campaign API probe."""
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import sys

from experiments.baselines.b4_embodiskill.state import write_json


def _memory():
    return {line.split(':')[0]: int(line.split(':')[1].split()[0])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}


def _hold(source, gamefile, connection):
    sys.path.insert(0, source)
    import gepa
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    env = None
    try:
        env = build_alfworld_env(env_num=1, eval_dataset="eval_in_distribution", seed=42,
                                specific_gamefiles=[gamefile])
        _, infos = env.reset({})
        if Path(infos[0]["extra.gamefile"]).resolve() != Path(gamefile).resolve():
            raise RuntimeError("GEPA load probe reset the wrong gamefile")
        connection.send({"ready": True, "pid": os.getpid()})
        connection.recv()
    except Exception as exc:
        connection.send({"ready": False, "error_type": type(exc).__name__})
        raise
    finally:
        if env is not None:
            env.close()


@contextmanager
def loaded_workers(source, gamefile, count, report_path):
    ctx = multiprocessing.get_context("spawn")
    processes, connections, samples = [], [], []
    initial = _memory()
    reserve = max(2*1024**3, int(initial["MemTotal"]*.15))
    try:
        for i in range(count):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_hold, args=(source,gamefile,child))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            if not parent.poll(300) or parent.recv().get("ready") is not True:
                raise RuntimeError("GEPA real dependency/ALFWorld load failed")
            now = _memory()
            samples.append(dict(workers=i+1, available=now["MemAvailable"]))
            per = max(1,(initial["MemAvailable"]-now["MemAvailable"])//(i+1))
            if now["MemAvailable"]-per*(count-i-1) < reserve:
                write_json(report_path, dict(passed=False, memory_load_passed=False,
                    target_workers=count, samples=samples, reserve_bytes=reserve))
                raise MemoryError("Protocol concurrency exceeds available ALFWorld worker memory")
        write_json(report_path, dict(passed=True, memory_load_passed=True, target_workers=count,
            loaded_workers=count, samples=samples, reserve_bytes=reserve))
        yield
    finally:
        for connection in connections:
            try:
                connection.send("release")
            except (OSError, EOFError):
                pass
            connection.close()
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join(15)


def main(argv=None):
    from experiments.baselines.common.provider_load_probe import main as provider_main
    from experiments.baselines.common.manifest import TaskManifestSet
    args = list(sys.argv[1:] if argv is None else argv)
    def arg(key):
        return args[args.index(key)+1]
    root = Path(__file__).resolve().parents[3]
    task = TaskManifestSet.load(root / 'data/baseline_manifests/validation_24.json').tasks[0]
    probe_dir = Path(arg('--output-dir'))
    memory_path = probe_dir.parent / (probe_dir.name+'_memory.json')
    try:
        with loaded_workers(arg('--skillopt-root'),str(Path(os.environ['ALFWORLD_DATA'])/task.gamefile_rel),
                            int(arg('--concurrency')),memory_path):
            return provider_main(args + ["--method", "b5_gepa"])
    except MemoryError:
        from experiments.baselines.b4_embodiskill.state import read_json
        report = read_json(memory_path)
        report.update(memory_load_report=str(memory_path))
        write_json(probe_dir / 'provider_load_probe.json', report)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
