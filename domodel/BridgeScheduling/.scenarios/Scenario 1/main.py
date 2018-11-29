from functools import partial, wraps
import os
from os.path import splitext
import time
import threading
import traceback
import sys

import pandas
from six import iteritems

from docplex.util.environment import get_environment

output_lock = threading.Lock()


def set_stop_callback(cb):
    env = get_environment()
    env.abort_callbacks += [cb]


def get_all_inputs():
    '''Utility method to read a list of files and return a tuple with all
    read data frames.
    Returns:
        a map { datasetname: data frame }
    '''
    result = {}
    env = get_environment()
    for iname in [f for f in os.listdir('.') if splitext(f)[1] == '.csv']:
        with env.get_input_stream(iname) as in_stream:
            df = pandas.read_csv(in_stream)
            datasetname, _ = splitext(iname)
            result[datasetname] = df
    return result


def callonce(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.called:
            wrapper.called = True
            return f(*args, **kwargs)
    wrapper.called = False
    return wrapper


@callonce
def write_all_outputs(outputs):
    '''Write all dataframes in ``outputs`` as .csv.

    Args:
        outputs: The map of outputs 'outputname' -> 'output df'
    '''
    global output_lock
    with output_lock:
        for (name, df) in iteritems(outputs):
            csv_file = '%s.csv' % name
            with get_environment().get_output_stream(csv_file) as fp:
                fp.write(df.to_csv(index=False))
    if len(outputs) == 0:
        print("Warning: no outputs written")


def wait_and_save_all_cb(outputs):
    global output_lock
    # just wait for the output_lock to be available
    t = time.time()
    with output_lock:
        pass
    elapsed = time.time() - t
    # write outputs
    write_all_outputs(outputs)


def get_line_of_model(n):
    env = get_environment()
    with env.get_input_stream('model.py') as m:
        lines = m.readlines()
        return lines[n - 1]


class InterpreterError(Exception):
    pass

if __name__ == '__main__':
    inputs = get_all_inputs()
    outputs = {}
    set_stop_callback(partial(wait_and_save_all_cb, outputs))

    env = get_environment()
    # The IS_DODS env must be True for model.py if running in DODS
    os.environ['IS_DODS'] = 'True'
    # This allows docplex.mp to behave the same (publish kpis.csv and solution.json
    # if this script is run locally
    os.environ['DOCPLEX_CONTEXT'] = 'solver.auto_publish=True'
    with env.get_input_stream('model.py') as m:
        try:
            exec(m.read().decode('utf-8'), globals())
        except SyntaxError as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            line_number = err.lineno
            imsg = 'File "model.py", line %s\n' % line_number
            imsg += err.text.rstrip() + '\n'
            spaces = ' ' * (err.offset - 1) if err.offset > 1 else ''
            imsg += spaces + "^\n"
            imsg += '%s: %s\n' % (error_class, detail)
            raise InterpreterError(imsg)
        except Exception as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            cl, exc, tb = sys.exc_info()
            ttb = traceback.extract_tb(tb)
            ttb[1] = ('model.py', ttb[1][1], ttb[1][2],
                      get_line_of_model(ttb[1][1]))
            line_number = ttb[1][1]
            ttb = ttb[1:]
            s = traceback.format_list(ttb)
            imsg = (''.join(s))
            imsg += '%s: %s\n' % (error_class, detail)
            raise InterpreterError(imsg)
        else:
            write_all_outputs(outputs)

