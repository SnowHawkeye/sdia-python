import numpy as np
import pytest

from sdia_python.lab2.box_window import BoxWindow, UnitBoxWindow


def test_raise_type_error_when_something_is_called():
    with pytest.raises(TypeError):
        # call_something_that_raises_TypeError()
        raise TypeError()


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[2.5, 2.5]]), "BoxWindow: [2.5, 2.5]"),
        (np.array([[0, 5], [0, 5]]), "BoxWindow: [0.0, 5.0] x [0.0, 5.0]"),
        (
            np.array([[0, 5], [-1.45, 3.14], [-10, 10]]),
            "BoxWindow: [0.0, 5.0] x [-1.45, 3.14] x [-10.0, 10.0]",
        ),
    ],
)
def test_box_string_representation(bounds, expected):
    assert str(BoxWindow(bounds)) == expected


@pytest.fixture
def box_2d_05():
    return BoxWindow(np.array([[0, 5], [0, 5]]))


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([[0, 0]]), True),
        (np.array([[2.5, 2.5], [0, 0]]), True),
        (np.array([[-1, 5]]), False),
        (np.array([[10, 3], [0, 0]]), False),
    ],
)
def test_indicator_function_box_2d(box_2d_05, point, expected):
    is_in = box_2d_05.indicator_function(point)
    assert is_in == expected


# ================================
# ==== WRITE YOUR TESTS BELOW ====
# ================================


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[2.5, 2.5]]), 1),
        (np.array([[0, 5], [0, 5]]), 2),
        (np.array([[0, 5], [-1.45, 3.14], [-10, 10]]), 3,),
    ],
)
def test_box_len(bounds, expected):
    assert len(BoxWindow(bounds)) == expected


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[2.5, 2.5]]), 1),
        (np.array([[0, 5], [0, 5]]), 2),
        (np.array([[0, 5], [-1.45, 3.14], [-10, 10]]), 3,),
    ],
)
def test_box_dimension(bounds, expected):
    assert BoxWindow(bounds).dimension() == expected


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[3, 6]]), 3),
        (np.array([[0, 5], [0, 5]]), 25),
        (np.array([[0, 5], [0.5, 1.5], [2, 3]]), 5),
    ],
)
def test_box_volume(bounds, expected):
    assert BoxWindow(bounds).volume() == expected


def test_raises_exception_when_point_of_wrong_dimension():
    with pytest.raises(ValueError):
        np.array([0, 0, 3]) in BoxWindow(np.array([[0, 5], [0, 5]]))


def test_raises_exception_when_initializing_with_wrong_segment_bounds():
    with pytest.raises(TypeError):
        BoxWindow(np.array([[5, 2], [0, 5]]))


def test_raises_exception_when_initializing_with_wrong_shape():
    with pytest.raises(TypeError):
        BoxWindow(np.array([[0, 5, 2], [0, 5, 3]]))


@pytest.mark.parametrize(
    "bounds, n",
    [
        (np.array([[3, 6]]), 5),
        (np.array([[0, 5], [0, 5]]), 3),
        (np.array([[0, 5], [0.5, 1.5], [2, 3]]), 10),
    ],
)
def test_random_points_generation_dimension(bounds, n):
    box = BoxWindow(bounds)
    points = box.rand(n)
    assert points.shape == (n, len(box))


# ================================
# ======== UNIT BOX TESTS ========
# ================================


def test_raises_exception_when_center_of_wrong_dimension():
    with pytest.raises(AssertionError):
        UnitBoxWindow(np.array([1, 1]), 1)


@pytest.mark.parametrize(
    "center, dimension, expected",
    [
        (None, 2, "BoxWindow: [-0.5, 0.5] x [-0.5, 0.5]"),
        (np.array([1, 2]), 2, "BoxWindow: [0.5, 1.5] x [1.5, 2.5]"),
    ],
)
def test_unit_box_initialization(center, dimension, expected):
    assert str(UnitBoxWindow(center, dimension)) == expected
