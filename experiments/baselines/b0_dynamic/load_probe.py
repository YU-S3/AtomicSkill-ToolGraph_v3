"""B0 real resident-memory gate; no optimizer or skill documents loaded."""
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import sys
import threading

from experiments.baselines.b4_embodiskill.state import write_json


def memory():
    return {s.split(':')[0]: int(s.split(':')[1].split()[0])*1024
            for s in Path('/proc/meminfo').read_text().splitlines()}


def hold(source, gamefile, connection):
    sys.path.insert(0, source)
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    env = None
    try:
        env = build_alfworld_env(env_num=1, eval_dataset="train", is_train=True,
                                seed=42, specific_gamefiles=[gamefile])
        _, infos = env.reset({})
        if Path(infos[0]["extra.gamefile"]).resolve() != Path(gamefile).resolve():
            raise RuntimeError("B0 load reset differs from exact gamefile")
        connection.send(True)
        connection.recv()
    finally:
        if env is not None:
            env.close()


@contextmanager
def loaded_workers(source, gamefile, count, report_path):
    ctx = multiprocessing.get_context("spawn")
    initial = memory()
    reserve = max(2*1024**3, int(initial["MemTotal"]*.15))
    report = dict(passed=False, target_workers=count, loaded_workers=0,
        memory_total=initial["MemTotal"], reserve_bytes=reserve,
        minimum_mem_available=initial["MemAvailable"], exact_gamefile_reset=False)
    processes, connections = [], []
    stop = threading.Event()
    def monitor():
        while not stop.wait(.25):
            report["minimum_mem_available"] = min(report["minimum_mem_available"], memory()["MemAvailable"])
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    body_passed = False
    try:
        if initial["MemAvailable"] <= reserve:
            raise MemoryError("Memory below reserve before B0 load")
        for index in range(count):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=hold, args=(source, gamefile, child))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            if not parent.poll(300) or parent.recv() is not True:
                raise RuntimeError("B0 resident ALFWorld initialization failed")
            report["loaded_workers"] = index+1
            available = memory()["MemAvailable"]
            report["minimum_mem_available"] = min(report["minimum_mem_available"], available)
            per = max(1, (initial["MemAvailable"]-available)//(index+1))
            if report["minimum_mem_available"] <= reserve or available-per*(count-index-1) <= reserve:
                raise MemoryError("B0 concurrency exceeds memory reserve")
        report["exact_gamefile_reset"] = True
        yield
        if not all(p.is_alive() for p in processes) or report["minimum_mem_available"] <= reserve:
            raise RuntimeError("Resident workers or memory reserve failed during provider load")
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
        report.update(worker_exit_codes=[p.exitcode for p in processes], forced_terminations=forced)
        report["passed"] = body_passed and not forced and all(p.exitcode==0 for p in processes)
        write_json(report_path, report)
        if body_passed and not report["passed"]:
            raise RuntimeError("B0 load workers did not release cleanly")
