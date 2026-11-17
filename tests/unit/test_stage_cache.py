"""
Tests for the caches of the stages of evaluations (see StageCache): a
stage is evaluated again only when the values it reads, or the stages it
takes, change; and an output that comes out the same as the last keeps
its version, so that the stages that take it are not evaluated again.
"""

from gbkfit.params import StageCache


def counted(output):
    """
    Return a stage of the given output (a function of the values it
    reads), and the values of each of its evaluations.
    """
    calls = []

    def stage(view):
        calls.append(dict(view))
        return output(view)
    return stage, calls


def test_a_stage_is_evaluated_again_when_what_it_reads_changes():
    first = StageCache(['a'])
    second = StageCache([], (first,))
    stage, calls = counted(lambda view: view['a'] * 2)
    taker, taken = counted(lambda view: None)
    for a, evaluations in ((1.0, 1), (1.0, 1), (2.0, 2)):
        assert first.evaluate(dict(a=a, b=0.0), stage) == 2 * a
        second.evaluate(dict(a=a), taker)
        assert len(calls) == len(taken) == evaluations
    assert first.version() == second.version() == 2


def test_an_output_the_same_as_the_last_keeps_its_version():
    # (an output of the same parity is the same: a step in what the stage
    # reads that leaves it so evaluates the stage, not those that take it)
    first = StageCache(['a'], same=lambda new, last: new % 2 == last % 2)
    second = StageCache([], (first,))
    stage, calls = counted(lambda view: int(view['a']))
    taker, taken = counted(lambda view: None)
    for a, output, version in ((1, 1, 1), (3, 1, 1), (4, 4, 2), (6, 4, 2)):
        assert first.evaluate(dict(a=a), stage) == output
        second.evaluate(dict(a=a), taker)
        assert first.version() == version
        assert len(taken) == version
    assert len(calls) == 4
    # (the values read are those of the last evaluation, kept or not)
    assert first.evaluate(dict(a=6), stage) == 4 and len(calls) == 4
    # (forgotten, its next output is new)
    first.forget()
    assert first.evaluate(dict(a=6), stage) == 6 and first.version() == 4
