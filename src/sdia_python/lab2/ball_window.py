import numpy as np

from sdia_python.lab2.utils import get_random_number_generator


class BallWindow:
    """Represents a ball in any dimension, defined by a center and a radius"""

    def __init__(self, center, radius):
        """
        Initializes a ball with a center and a radius. The radius must be positive.

        Args:
            center (numpy.array): coordinates of the center point
            radius (float): float representing the radius
        """
        center = np.array(center)
        assert len(center)
        assert radius >= 0

        self.center = center
        self.radius = float(radius)

    def __str__(self):
        return f"BallWindow: ({str(self.center)}, {str(self.radius)})"

    def __len__(self):
        """Returns the dimension of the ball

        Returns:
            int : dimension of the ball
        """
        return len(self.center)

    def __contains__(self, point):
        """Tells whether a point is contained in the ball

        Args:
            point (numpy.array): the point to test

        Returns:
            boolean: if it is contained in the ball
        """
        point = np.array(point)
        assert self.dimension() == point.size

        return np.linalg.norm(point - self.center) <= self.radius

    def dimension(self):
        """Returns the dimension of the ball

        Returns:
            int : dimension of the ball
        """
        return len(self)

    def volume(self):
        """Returns the volume of the ball. The formula for the volume of an n-ball is used : https://fr.wikipedia.org/wiki/N-sph%C3%A8re

        Returns:
            float : volume of the ball
        """
        n = self.dimension()
        R = self.radius
        if n % 2 == 0:  # formula in case dimension is even
            return (((np.pi) ** (n / 2)) * R ** n) / np.math.factorial(n / 2)
        else:  # formula in case dimension is odd
            odds = [i for i in range(1, n + 1, 2)]
            product = np.product(odds)
            return 2 ** ((n + 1) / 2) * np.pi ** ((n - 1) / 2) * R ** n / product

    def indicator_function(self, points):
        """Returns true if all points are in the ball.

        Args:
            points (np.array): Array of points to test

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
            numpy.array: array containing the n generated points
        """
        rng = get_random_number_generator(rng)

        directions = rng.uniform(size=(n, self.dimension()))
        directions = np.array(
            [direction / np.linalg.norm(direction) for direction in directions]
        )
        distances = rng.uniform(0, self.radius, n)
        vectors = np.array(
            [
                direction * distance
                for (direction, distance) in zip(directions, distances)
            ]
        )
        return np.array([vector + self.center for vector in vectors])
