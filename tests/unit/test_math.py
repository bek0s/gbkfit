from gbkfit.math.math import *


def test_is_even_is_odd():
    for num in [0, 0.0, 2, 2.0, 2.5]:
        assert is_even(+num)
        assert is_even(-num)
        assert not is_odd(+num)
        assert not is_odd(-num)
    for num in [1, 1.0, 1.5]:
        assert not is_even(+num)
        assert not is_even(-num)
        assert is_odd(+num)
        assert is_odd(-num)


def test_round_functions():
    # round even
    assert roundd_even(0.0) == 0
    assert roundd_even(+3.0) == +2
    assert roundd_even(-3.0) == -4
    assert roundd_even(+3.5) == +2
    assert roundd_even(-3.5) == -4
    assert roundu_even(0.0) == 0
    assert roundu_even(+3.0) == +4
    assert roundu_even(-3.0) == -2
    assert roundu_even(+3.5) == +4
    assert roundu_even(-3.5) == -2
    # round odd
    assert roundd_odd(0.0) == -1
    assert roundd_odd(+2.0) == +1
    assert roundd_odd(-2.0) == -3
    assert roundd_odd(+2.5) == +1
    assert roundd_odd(-2.5) == -3
    assert roundu_odd(0.0) == 1
    assert roundu_odd(+2.0) == +3
    assert roundu_odd(-2.0) == -1
    assert roundu_odd(+2.5) == +3
    assert roundu_odd(-2.5) == -1
    # round multiple
    assert roundd_multiple(+1.0, 2.0) == 0
    assert roundd_multiple(+5.0, 2.0) == +4
    assert roundd_multiple(-1.0, 2.0) == -2
    assert roundd_multiple(-5.0, 2.0) == -6
    assert roundu_multiple(+1.0, 2.0) == +2
    assert roundu_multiple(+5.0, 2.0) == +6
    assert roundu_multiple(-1.0, 2.0) == 0
    assert roundu_multiple(-5.0, 2.0) == -4
    # round power of two
    assert roundd_po2(3.0) == 2
    assert roundd_po2(3.5) == 2
    assert roundu_po2(3.0) == 4
    assert roundu_po2(3.5) == 4


def test_foo():

    assert uniform_1d_fun(0, 10, -10, 10) == 10
    assert uniform_1d_fun(-11, 10, -10, 10) == 0
    assert uniform_1d_fun(11, 10, -10, 10) == 0

    assert uniform_1d_cdf(0, -10, 10) == 0.5
    assert uniform_1d_cdf(-10, -10, 10) == 0
    assert uniform_1d_cdf(10, -10, 10) == 1

