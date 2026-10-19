"""Business stdout travels as a bounded result, independently of worker diagnostics."""
import inspect

STDOUT_MAX_BYTES = 65536


class ResultWriter:
    def __init__(self, limit):
        self.limit, self.parts, self.size, self.total = limit, [], 0, 0

    def write(self, text):
        encoded=text.encode('utf-8');self.total+=len(encoded)
        kept=encoded[:max(0,self.limit-self.size)].decode('utf-8',errors='ignore')
        if kept:self.parts.append(kept)
        self.size+=len(kept.encode('utf-8'))
        return len(text)

    def flush(self):pass


def source_wrapper(source):
    # Self-contained source runs inside the existing pinned Docker sandbox.
    return inspect.getsource(ResultWriter)+'''
def run(ctx, inputs):
    import contextlib, json
    writer = ResultWriter('''+str(STDOUT_MAX_BYTES)+''')
    namespace = {"ctx": ctx, "INPUT_PATH": "/workspace/inputs/input.xlsx", "OUTPUT_PATH": "/workspace/case1_result.xlsx"}
    with contextlib.redirect_stdout(writer):
        exec('''+repr(source)+''', namespace)
    stdout = ''.join(writer.parts)
    outputs = {"files": inputs["files"], "deleted_files": inputs["deleted_files"],
        "stdout": stdout, "stdout_receipt": {"total_bytes": writer.total, "retained_bytes": writer.size,
            "limit_bytes": writer.limit, "truncated": writer.total > writer.size}}
    if writer.total == writer.size:
        try:
            computed = json.loads(stdout)
            json.dumps(computed, allow_nan=False)
            if isinstance(computed, dict): outputs["computed"] = computed
        except (ValueError, TypeError): pass
    return {"status": "ok", "outputs": outputs}
'''
