"""Execute a notebook in-process with IPython (no kernel / zmq) and save outputs into it.
Usage: python run_inprocess.py notebook.ipynb [logfile]"""
import sys
import time
import traceback

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path = sys.argv[1]
log = open(sys.argv[2], "a", encoding="utf-8") if len(sys.argv) > 2 else sys.stdout


def say(msg):
    log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    log.flush()


shell = InteractiveShell.instance()
shell.enable_gui = lambda gui=None: None
shell.run_line_magic("matplotlib", "inline")

results = []
shell.displayhook.write_output_prompt = lambda: None
shell.displayhook.write_format_data = lambda data, md=None: results.append((data, md or {}))

nb = nbformat.read(path, as_version=4)
count = 0
for index, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    count += 1
    results.clear()
    start = time.time()
    say(f"cell {index} start")
    with capture_output() as cap:
        res = shell.run_cell(cell.source, store_history=True)
    outputs = []
    if cap.stdout:
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=cap.stdout))
    if cap.stderr:
        outputs.append(nbformat.v4.new_output("stream", name="stderr", text=cap.stderr))
    for out in cap.outputs:
        outputs.append(nbformat.v4.new_output("display_data", data=out.data, metadata=out.metadata))
    for data, md in results:
        outputs.append(nbformat.v4.new_output("execute_result", data=data, metadata=md, execution_count=count))
    error = res.error_in_exec or res.error_before_exec
    if error is not None:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=tb))
    cell.outputs = outputs
    cell.execution_count = count
    nbformat.write(nb, path)
    say(f"cell {index} done in {time.time() - start:.1f}s" + (f"  ERROR {type(error).__name__}: {error}" if error else ""))
    if error is not None:
        say("stopped on error")
        sys.exit(1)
say("finished")
