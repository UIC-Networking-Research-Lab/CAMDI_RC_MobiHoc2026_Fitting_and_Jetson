"""Small command dispatchers; experiment parsers retain their existing arguments."""
import argparse
import importlib
import sys


def _dispatch(commands, description, argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=list(commands), help="Use COMMAND --help for its options.")
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(arguments[:1]).command
    module = importlib.import_module(commands[command])
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0] + " " + command, *arguments[1:]]
        return module.main()
    finally:
        sys.argv = saved_argv


def run(argv=None):
    return _dispatch({
        "resnet": "jetson_inference.resnet.cli",
        "flan-t5": "jetson_inference.flan_t5.cli",
        "multi": "jetson_inference.multi_task.cli",
    }, "Run a four-node online inference experiment.", argv)


def prepare(argv=None):
    return _dispatch({
        "assets": "jetson_inference.assets",
        "dataset": "jetson_inference.dataset_export",
    }, "Validate/download model assets or export the SST-2 dataset.", argv)


def plot(argv=None):
    return _dispatch({
        "single": "jetson_inference.analysis.single_task",
        "multi": "jetson_inference.analysis.multi_task",
    }, "Plot saved single-task or multi-task experiment results.", argv)
