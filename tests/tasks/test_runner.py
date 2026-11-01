"""
Tests for the runner of the tasks (python -m gbkfit.tasks): it runs a
task with its options as JSON, and rejects the options that are not the
task's.
"""

import json
import os
import subprocess
import sys

import pytest

from gbkfit.tasks import TASKS, prep


def run_task(*args):
    """Run the runner with the given arguments."""
    return subprocess.run(
        [sys.executable, '-m', 'gbkfit.tasks', *args],
        capture_output=True, text=True)


def test_a_task_runs_with_its_options():
    options = dict(output_dir='tables', output_dir_mode='terminate')
    result = run_task('prep_lsf_muse', json.dumps(options))
    assert result.returncode == 0, result.stderr
    assert os.path.isfile('tables/prep_lsf.ecsv')


@pytest.mark.parametrize('args, status, message', [
    # (options that are not the task's are an error of the user's input,
    # as one of a configuration; an invalid command line is argparse's)
    (['prep_lsf_muse', '{"directory": "x"}'], 1,
     "unexpected keyword argument 'directory'"),
    (['eval', '{}'], 1, "missing a required argument: 'mode'"),
    (['prep_lsf_muse', '[]'], 2, "must be a JSON object"),
    (['evaluate'], 2, "invalid choice: 'evaluate'")])
def test_invalid_tasks_and_options_are_rejected(args, status, message):
    result = run_task(*args)
    assert result.returncode == status and message in result.stderr


def test_every_prep_task_is_a_task():
    assert {name for name in TASKS if name.startswith('prep_')} == set(
        prep.__all__)
