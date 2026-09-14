"""Hold real GEPA/SkillOpt/ALFWorld workers during the campaign API probe."""
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import sys
import threading

from experiments.baselines.b4_embodiskill.state import write_json


def _memory():
    return {line.split(':')[0]: int(line.split(':')[1].split()[0])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}


def _hold(source, gamefile, connection):
    sys.path.insert(0, source)
    import gepa
    import skillopt
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    env = None
    try:
        Path(gepa.__file__).resolve().relative_to(Path(source).parent / "gepa")
        Path(skillopt.__file__).resolve().relative_to(Path(source).resolve())
        env = build_alfworld_env(env_num=1, eval_dataset="eval_in_distribution", seed=42,
                                specific_gamefiles=[gamefile])
        _, infos = env.reset({})
        if Path(infos[0]["extra.gamefile"]).resolve() != Path(gamefile).resolve():
            raise RuntimeError("GEPA load probe reset the wrong gamefile")
        connection.send({"ready": True, "pid": os.getpid(), "exact_gamefile_reset": True,
                         "gepa_origin": gepa.__file__, "skillopt_origin": skillopt.__file__})
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
    report = dict(passed=False, memory_load_passed=False, target_workers=count,
                  memory_total=initial["MemTotal"], reserve_bytes=reserve,
                  minimum_mem_available=initial["MemAvailable"], samples=samples,
                  workers=[], exact_gamefile=str(Path(gamefile).resolve()),
                  exact_gamefile_reset=False, loaded_workers=0)
    stop = threading.Event()
    def monitor():
        while not stop.wait(.25):
            report["minimum_mem_available"] = min(report["minimum_mem_available"], _memory()["MemAvailable"])
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    body_passed = False
    try:
        if initial["MemAvailable"] <= reserve:
            raise MemoryError("Initial memory below frozen reserve")
        for i in range(count):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_hold, args=(source,gamefile,child))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            if not parent.poll(300):
                raise RuntimeError("GEPA real dependency/ALFWorld load failed")
            ready = parent.recv()
            if ready.get("ready") is not True:
                raise RuntimeError("GEPA real dependency/ALFWorld load failed")
            report["workers"].append(ready)
            report["loaded_workers"] = i+1
            now = _memory()
            samples.append(dict(workers=i+1, available=now["MemAvailable"]))
            report["minimum_mem_available"] = min(report["minimum_mem_available"], now["MemAvailable"])
            per = max(1,(initial["MemAvailable"]-now["MemAvailable"])//(i+1))
            if report["minimum_mem_available"] <= reserve or now["MemAvailable"]-per*(count-i-1) <= reserve:
                raise MemoryError("Protocol concurrency exceeds available ALFWorld worker memory")
        report["exact_gamefile_reset"] = all(w["exact_gamefile_reset"] for w in report["workers"])
        yield
        if not all(p.is_alive() for p in processes):
            raise RuntimeError("Resident worker crashed during provider probe")
        if report["minimum_mem_available"] <= reserve:
            raise MemoryError("Resident memory dropped below reserve during provider probe")
        body_passed = True
    except Exception as exc:
        report.update(error_type=type(exc).__name__, memory_rejected=isinstance(exc, MemoryError))
        raise
    finally:
        for connection in connections:
            try:
                connection.send("release")
            except (OSError, EOFError):
                pass
            connection.close()
        forced = []
        for process in processes:
            process.join(15)
            if process.is_alive():
                forced.append(process.pid)
                process.terminate()
                process.join(15)
        stop.set()
        watcher.join(5)
        report.update(worker_exit_codes=[p.exitcode for p in processes], forced_terminations=forced,
                      workers_released=not forced and all(p.exitcode == 0 for p in processes))
        passed = body_passed and report["workers_released"] and report["minimum_mem_available"] > reserve
        report.update(passed=passed, memory_load_passed=passed)
        write_json(report_path, report)
        if body_passed and not passed:
            raise RuntimeError("Load probe worker release or memory reserve failed")


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
            provider_rc = provider_main(args + ["--method", "b5_gepa"])
    except Exception:
        provider_rc = 1
    from experiments.baselines.b4_embodiskill.state import read_json
    memory = read_json(memory_path)
    path = probe_dir / 'provider_load_probe.json'
    report = read_json(path) if path.is_file() else {"passed": False}
    report.update(memory_load_report=str(memory_path), memory_load_passed=memory["passed"],
                  memory_rejected=memory.get("memory_rejected", False))
    report["passed"] = report.get("passed") is True and memory["passed"] and provider_rc == 0
    write_json(path, report)
    return 0 if report["passed"] else 1


if __name__ == '__main__':
    raise SystemExit(main())
