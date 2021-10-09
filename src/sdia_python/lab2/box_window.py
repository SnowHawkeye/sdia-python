import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BoxWindow:
    """Represents a box in any dimension, defined by a number of segments."""

    def __init__(self, bounds):
        """Initializes the bounds with the np.array given as a parameter. Bounds must be given as arrays of [a, b] with a <= b.

        Args:
            bounds (np.array): Array containing the bounds for each dimension.
        """

        bounds = np.array(bounds)
        if not bounds.shape[1] == 2:
            raise TypeError("Not a segment")

        if not np.all([segment[1] - segment[0] >= 0 for segment in bounds]):
            raise TypeError("Segment bounds are not in the right order")

        self.bounds = bounds

    def __str__(self):
        r"""BoxWindow: :math:`[a_1, b_1] \times [a_2, b_2] \times \cdots`

        Returns:
            str: = Same string returned as when calling print(BoxWindow(...)).
        """

        print_segment = (
            lambda segment: f"[{str(float(segment[0]))}, {str(float(segment[1]))}]"
        )
        string = ""
        for i, segment in enumerate(self.bounds):
            string += print_segment(segment)
            if i != (len(self.bounds) - 1):
                string += " x "
        result = "BoxWindow: " + string
        return result

    def __len__(self):
        """Returns the dimension of the box.

        Returns:
            int: Dimension of the box.
        """
        return len(self.bounds)

    def __contains__(self, point):
        """Returns True if a point is in the box.

        Args:
            point (np.array): The point to test.

        Returns:
            bool: True if ``point`` is contained in this `BoxWindow`.
        """
        if len(point) != len(self):
            raise ValueError("Wrong dimension of point")
        return all(a <= x <= b for (a, b), x in zip(self.bounds, point))

    def dimension(self):
        """Returns the dimension of the box.

        Returns:
            int: Dimension of the box.
        """
        return len(self)

    def volume(self):
        """Returns the volume of the box.

        Returns:
            int: Volume of the box.
        """

        volume = 0 if len(self) == 0 else 1
        volume = np.prod(np.diff(self.bounds, axis=1))

        return volume

    def indicator_function(self, points):
        """Returns ``true`` if all points are in the box.

        Args:
            points (np.array): Array of points to test.

        Returns:
            bool: True if all points are in the box.
        """

        return np.all([point in self for point in points])

    def rand(self, n=1, rng=None):
        """Generates ``n`` points uniformly at random inside the :py:class:`BoxWindow`.

        Args:
            n (int, optional): [description]. Defaults to 1.
            rng ([type], optional): [description]. Defaults to None.

        Returns:
            numpy.array: array containing all the generated points
        """
        rng = get_random_number_generator(rng)
        values = np.array(
            [rng.uniform(segment[0], segment[1], n) for segment in self.bounds]
        )
        return values.T


class UnitBoxWindow(BoxWindow):
    def __init__(self, center, dimension):
        """Represents a box in any dimension, where all segments defining the box are of length 1, centered on  ``center`` .

        Args:
            dimension (int): [description] Dimension of the box.
            center (numpy.array, optional): Center of the segments. Defaults to None.
        """

        if center is None:
            center = np.zeros(dimension)
        else:
            assert len(center) == dimension

        segments = np.full(shape=(2, dimension), fill_value=[-0.5, 0.5])
        bounds = (segments.T + center).T
        super(UnitBoxWindow, self).__init__(bounds)
