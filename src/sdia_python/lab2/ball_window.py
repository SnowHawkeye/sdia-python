import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BallWindow:
    """Represents a ball in any dimension, defined by a center and a radius."""

    def __init__(self, center, radius):
        """Initializes a ball with a center and a radius. The radius must be positive.

        Args:
            center (numpy.array): Coordinates of the center point.
            radius (float): Float representing the radius.
        """
        center = np.array(center)
        assert len(center)
        assert radius >= 0

        self.center = center
        self.radius = float(radius)

    def __str__(self):
        return f"BallWindow: ({str(self.center)}, {str(self.radius)})"

    def __len__(self):
        """Returns the dimension of the ball.

        Returns:
            int : Dimension of the ball.
        """
        return len(self.center)

    def __contains__(self, point):
        """Returns ``true`` if the given `point` is contained in the ball.

        Args:
            point (numpy.array): The point to test.

        Returns:
            boolean: True if ``point`` is contained in this ``BallWindow``.
        """
        point = np.array(point)
        assert self.dimension() == point.size

        return np.linalg.norm(point - self.center) <= self.radius

    def dimension(self):
        """Returns the dimension of the ball.

        Returns:
            int : Dimension of the ball.
        """
        return len(self)

    def volume(self):
        """Returns the volume of the ball. The formula for the volume of an n-ball is used : https://fr.wikipedia.org/wiki/N-sph%C3%A8re.

        Returns:
            float : Volume of the ball.
        """
        n = self.dimension()
        R = self.radius
        if n % 2 == 0:  # formula in case dimension is even
            return (((np.pi) ** (n / 2)) * R ** n) / np.math.factorial(n / 2)
        else:  # formula in case dimension is odd
            odds = np.arange(1, n + 1, 2)
            product = np.product(odds)
            return 2 ** ((n + 1) / 2) * np.pi ** ((n - 1) / 2) * R ** n / product

    def indicator_function(self, points):
        """Returns ``true`` if all points are in the ball.

        Args:
            points (np.array): Array of points to test.

        Returns:
            bool: True if all points are in the box.
        """
        return np.all([point in self for point in points])

    def rand(self, n=1, rng=None):
        """Generates n points in the ball.

        Args:
            n (int, optional): Number of points generated. Defaults to 1.
            rng (int, optional): Random seed. Defaults to None.

        Returns:
            numpy.array: Array containing the ``n`` generated points.
        """
        rng = get_random_number_generator(rng)

        directions = rng.uniform(size=(n, self.dimension()))
        directions = np.array(
            [direction / np.linalg.norm(direction) for direction in directions]
        )
        # for the direction to follow a uniform distribution, it is the square root of the radius that must be uniformly distributed
        distances = rng.uniform(0, np.sqrt(self.radius), n)
        vectors = np.array(
            [
                direction * distance ** 2
                for (direction, distance) in zip(directions, distances)
            ]
        )
        return vectors + self.center


class UnitBallWindow(BallWindow):
    def __init__(self, center, dimension):
        """Represents a ball in any dimension, of radius 1, of center  ``center`` .

        Args:
            dimension (int): [description] Dimension of the ball.
            center (numpy.array, optional): Center of the ball. Defaults to None.
        """

        if center is None:
            center = np.zeros(dimension)
        else:
            assert len(center) == dimension
        super(UnitBallWindow, self).__init__(center, 1)
