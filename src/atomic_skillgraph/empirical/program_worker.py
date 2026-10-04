"""Linux namespace sandbox with a bounded, permission-checked JSON RPC broker.

No user source is executed in the parent. AST checks are not the sandbox.
"""
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def _kernel_limits(memory_mb, seconds):
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(seconds), math.ceil(seconds) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Namespaces hide host resources. Seccomp additionally prevents descendants,
    # new namespaces, network sockets and replacement executables.
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes, lib.seccomp_init.restype = [ctypes.c_uint32], ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in ("fork", "vfork", "clone", "clone3", "execve", "execveat", "socket", "connect",
                     "ptrace", "mount", "umount2", "unshare", "setns", "kill", "tgkill"):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("seccomp rule failed")
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError("seccomp load failed")
    finally:
        lib.seccomp_release(ctx)


def _worker():
    original_out, original_in = sys.stdout, sys.stdin
    limit = int(sys.argv[4])
    def send(value):
        encoded = json.dumps(value, allow_nan=False)
        if len(encoded.encode()) > limit:
            raise ValueError("RPC message limit exceeded")
        original_out.write(encoded + "\n")
        original_out.flush()

    def rpc(method, **values):
        send({"kind": "rpc", "method": method, **values})
        raw = original_in.buffer.readline(limit + 1)
        if len(raw) > limit or not raw.endswith(b"\n"):
            raise ValueError("Invalid broker response")
        response = json.loads(raw)
        if "rpc_error" in response:
            raise RuntimeError(response["rpc_error"])
        return response["value"]

    class Context:
        def observe(self):
            return rpc("observe")

        def available_tools(self):
            return rpc("available_tools")

        def remaining_calls(self):
            return rpc("remaining_calls")

        def call(self, name, arguments):
            return rpc("call", name=name, arguments=arguments)

    source = Path("/program/source.py").read_text()
    inputs = json.loads(Path("/program/inputs.json").read_text())
    _kernel_limits(int(sys.argv[2]), float(sys.argv[3]))
    # User prints are diagnostics, never broker requests. The process cannot
    # read the broker's host files or environment through either stream.
    sys.stdout = sys.stderr
    try:
        namespace = {"__name__": "generated_program"}
        exec(compile(source, "/program/source.py", "exec"), namespace)
        value = namespace["run"](Context(), inputs)
        if not isinstance(value, dict) or value.get("status") not in {"ok", "not_found", "needs_input", "blocked"}:
            raise ValueError("run must return a dictionary with a valid status")
        if not isinstance(value.get("outputs", {}), dict):
            raise ValueError("outputs must be a dictionary")
        send({"kind": "done", "result": value})
    except BaseException as exc:
        send({"kind": "done", "result": {"status": "execution_error", "error": str(exc)[:2048]}})


class ProgramWorker:
    def __init__(self, settings=None):
        self.settings = {"max_tool_calls_per_invocation": 32, "wall_timeout_seconds": 60,
                         "memory_limit_mb": 1024, "max_rpc_message_bytes": 1048576, **(settings or {})}
        self.invocations = []

    def execute(self, program, inputs, broker):
        from .contracts import validate_schema_instance
        validate_schema_instance(inputs, program["input_schema"])
        if sys.platform != "linux" or not shutil.which("bwrap"):
            raise RuntimeError("sandbox_python_v1 requires Linux bubblewrap; no unsafe fallback")
        source = program["source"]
        started = time.monotonic()
        calls, result = 0, {"status": "execution_error", "error": "worker exited without a result"}
        s = self.settings
        with tempfile.TemporaryDirectory(prefix="asg-program-") as directory:
            root = Path(directory)
            (root / "source.py").write_text(source, encoding="utf-8")
            (root / "inputs.json").write_text(json.dumps(inputs, allow_nan=False), encoding="utf-8")
            command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                       "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                       "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                       "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--clearenv",
                       "--ro-bind", str(root), "/program", "--ro-bind", str(Path(__file__).resolve()), "/worker.py",
                       "--chdir", "/tmp", "/usr/bin/python3", "-I", "-u", "/worker.py", "--worker",
                       str(s["memory_limit_mb"]), str(s["wall_timeout_seconds"]), str(s["max_rpc_message_bytes"])]
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=errors, start_new_session=True, env={"PATH": "/usr/bin:/bin"})
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                buffer, finished = b"", False
                try:
                    while not finished:
                        remaining = s["wall_timeout_seconds"] - (time.monotonic() - started)
                        if remaining <= 0:
                            result = {"status": "execution_error", "error": "worker wall timeout"}
                            break
                        if not selector.select(min(remaining, .2)):
                            if process.poll() is not None:
                                break
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        buffer += chunk
                        if len(buffer) > s["max_rpc_message_bytes"]:
                            raise ValueError("worker RPC message limit exceeded")
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            request = json.loads(line)
                            if request.get("kind") == "done":
                                result, finished = request["result"], True
                                break
                            method = request.get("method")
                            if method == "observe":
                                value = broker.observe()
                            elif method == "available_tools":
                                value = [t for t in broker.available_tools() if t["name"] in program["allowed_tools"]]
                            elif method == "remaining_calls":
                                value = min(broker.remaining_calls(), s["max_tool_calls_per_invocation"] - calls)
                            elif method == "call":
                                if request.get("name") not in program["allowed_tools"]:
                                    raise PermissionError("Program tool is not authorized")
                                if calls >= s["max_tool_calls_per_invocation"] or broker.remaining_calls() <= 0:
                                    raise RuntimeError("Program/shared task tool budget exhausted")
                                calls += 1
                                value = broker.call(request["name"], request.get("arguments"))
                            else:
                                raise ValueError("Unknown broker RPC")
                            response = json.dumps({"value": value}, allow_nan=False).encode() + b"\n"
                            if len(response) > s["max_rpc_message_bytes"]:
                                raise ValueError("broker result must use a resource handle")
                            process.stdin.write(response)
                            process.stdin.flush()
                except (ValueError, OSError, RuntimeError, PermissionError) as exc:
                    result = {"status": "execution_error", "error": str(exc)}
                finally:
                    selector.close()
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    errors.seek(0)
                    diagnostic = errors.read(4096).decode(errors="replace")
        if not isinstance(result, dict) or result.get("status") not in {
                "ok", "not_found", "needs_input", "blocked", "execution_error"}:
            result = {"status": "execution_error", "error": "Invalid worker result"}
        if result["status"] == "ok":
            try:
                validate_schema_instance(result.get("outputs"), program["output_schema"])
            except ValueError as exc:
                result = {"status": "execution_error", "error": str(exc)}
        result = {**result, "calls": calls, "elapsed_seconds": time.monotonic() - started,
                  "diagnostic": diagnostic, "program_id": program["id"]}
        self.invocations.append(result)
        return result


if __name__ == "__main__" and sys.argv[1:2] == ["--worker"]:
    _worker()
