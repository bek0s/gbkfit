"""
Tests of the non-uniform FFTs of the drivers, against direct sums.
"""

import numpy as np
import pytest


# The tolerance asked of the NUFFTs, and the relative error allowed of
# them (of the 2-norm)
EPS, ERROR = 1e-12, 1e-11


def transform(driver, nufft, strengths, shape):
    out = driver.mem_alloc_d(shape, np.complex128)
    nufft(driver.mem_copy_h2d(strengths), out)
    return driver.mem_copy_d2h(out)


def relative_error(result, expected):
    return np.linalg.norm(result - expected) / np.linalg.norm(expected)


def test_type3_is_the_direct_sum(driver):
    # Phases of up to 60 rad along each axis
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, (300, 3)) * [0.001, 0.001, 0.0005]
    frequencies = rng.uniform(-1, 1, (200, 3)) * 1e4
    strengths = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    plan = driver.nufft().type3(points, frequencies, -1, EPS)
    result = transform(driver, plan, strengths, (200,))
    expected = np.exp(-2j * np.pi * frequencies @ points.T) @ strengths
    assert relative_error(result, expected) < ERROR


def test_type1_is_the_direct_sum(driver):
    # An image of 5 x 8 pixels (odd and even axes): f[q] at the modes
    # k = q - shape // 2, a column of the points for each axis
    rng = np.random.default_rng(2)
    shape = (5, 8)
    points = rng.uniform(-0.5, 0.5, (300, 2))
    strengths = rng.standard_normal(300) + 1j * rng.standard_normal(300)
    plan = driver.nufft().type1(points, shape, 1, EPS)
    result = transform(driver, plan, strengths, shape)
    ky, kx = np.indices(shape) - np.array(shape)[:, None, None] // 2
    expected = np.exp(2j * np.pi * (
        ky[..., None] * points[:, 0] + kx[..., None] * points[:, 1])) \
        @ strengths
    assert relative_error(result, expected) < ERROR


@pytest.mark.parametrize('arguments, message', [
    (dict(points=np.zeros((4, 4))), "shape \\(n, d\\)"),
    (dict(frequencies=np.zeros((4, 2))), "dimensions of the points"),
    (dict(sign=0), "sign"),
    (dict(eps=0), "eps")])
def test_type3_arguments_are_checked(driver, arguments, message):
    arguments = dict(
        points=np.zeros((4, 3)), frequencies=np.zeros((4, 3)), sign=-1,
        eps=EPS) | arguments
    with pytest.raises(ValueError, match=message):
        driver.nufft().type3(**arguments)


def test_type1_needs_an_axis_for_each_dimension(driver):
    with pytest.raises(ValueError, match="an axis for each dimension"):
        driver.nufft().type1(np.zeros((4, 2)), (8,), 1, EPS)
