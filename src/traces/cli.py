"""Collect four-node performance traces using the task's original options."""
import argparse
import importlib
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='trace.py', description=__doc__)
    parser.add_argument('task', choices=['resnet', 'flan-t5'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.parse_args(argv)
    route = parser.parse_args(argv[:1])
    module = importlib.import_module('traces.performance_' + route.task.replace('-', '_'))
    saved_argv = sys.argv
    try:
        sys.argv = ['trace.py ' + route.task] + argv[1:]
        return module.main()
    finally:
        sys.argv = saved_argv


if __name__ == '__main__':
    main()
