import numpy as np
import pytest

from sdia_python.lab2.ball_window import BallWindow


def test_raises_exception_when_initializing_with_wrong_center():
    with pytest.raises(AssertionError):
        BallWindow(center=np.array([]), radius=5)


def test_raises_exception_when_initializing_with_wrong_radius():
    with pytest.raises(AssertionError):
        BallWindow(center=np.array([1, 1]), radius=-3)


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([0]), 5, 1),
        (np.array([0, 0]), 5, 2),
        (np.array([0, 0, 0]), 5, 3),
        (np.array([0, 0, 0, 0]), 5, 4),
    ],
)
def test_dimension(center, radius, expected):
    assert BallWindow(center, radius).dimension() == expected


def test_raises_exception_when_using_contains_with_invalid_dimension():
    with pytest.raises(AssertionError):
        ball = BallWindow(center=np.array([0, 0]), radius=5)
        x = np.array([1, 2, 3])
        x in ball


@pytest.fixture
def ball_2d():
    return BallWindow(center=np.array([0, 0]), radius=5)


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([[0, 0]]), True),
        (np.array([[0, 0], [2.5, 2.5]]), True),
        (np.array([[0, 30]]), False),
        (np.array([[0, 0], [-6, 9]]), False),
    ],
)
def test_indicator_function2D(ball_2d, point, expected):
    is_in = ball_2d.indicator_function(point)
    assert is_in == expected


@pytest.fixture
def ball_1d():
    return BallWindow(center=np.array([5]), radius=2)


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([[6]]), True),
        (np.array([[4.5], [6]]), True),
        (np.array([[1], [6]]), False),
    ],
)
def test_indicator_function1D(ball_1d, point, expected):
    is_in = ball_1d.indicator_function(point)
    assert is_in == expected


@pytest.mark.parametrize(
    "center, radius, n",
    [
        (np.array([3, 6]), 5, 5),
        (np.array([0, 5, 8]), 3, 10),
        (np.array([0.5, 1.5, 3.5, 7]), 10, 15),
    ],
)
def test_random_points_generation_dimension(center, radius, n):
    ball = BallWindow(center, radius)
    points = ball.rand(n)
    assert points.shape == (n, len(ball))


@pytest.mark.parametrize(
    "center, radius, n",
    [
        (np.array([3, 6]), 5, 5),
        (np.array([0, 5, 8]), 3, 10),
        (np.array([0.5, 1.5, 3.5, 7]), 10, 15),
    ],
)
def test_random_points_generation_contained(center, radius, n):
    ball = BallWindow(center, radius)
    points = ball.rand(n)
    assert np.all([point in ball for point in points])


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([0]), 2, 4),
        (np.array([0, 0]), 2, np.pi * 4),
        (np.array([0, 0, 0]), 2, np.pi * 2 ** 3 * 4 / 3),
    ],
)
def test_volume(center, radius, expected):
    assert BallWindow(center, radius).volume() == expected


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([2.5, 2.5]), 5, "BallWindow: ([2.5 2.5], 5.0)"),
        (np.array([0, 5, 2]), 3, "BallWindow: ([0 5 2], 3.0)"),
    ],
)
def test_ball_string_representation(center, radius, expected):
    assert str(BallWindow(center, radius)) == expected
