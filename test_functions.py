import unittest
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Assuming the original code is imported and available under the same module
# For this example, I will redefine the necessary classes and functions within this code block.


@dataclass
class FunctionData:
    """Data class to store function details."""

    name: str
    domain: tuple


class Function(ABC):
    """Abstract base class for mathematical functions."""

    def __init__(self, data: FunctionData):
        self.data = data

    @abstractmethod
    def evaluate(self, x):
        """Evaluate the function at x."""
        pass

    def is_concave(self):
        """Check if the function is concave."""
        x = np.linspace(*self.data.domain, 1000)
        second_derivative = np.gradient(np.gradient(self.evaluate(x)))
        return np.all(second_derivative <= 0)

    def is_convex(self):
        """Check if the function is convex."""
        x = np.linspace(*self.data.domain, 1000)
        second_derivative = np.gradient(np.gradient(self.evaluate(x)))
        return np.all(second_derivative >= 0)


class QuadraticFunction(Function):
    """Concrete class for quadratic functions."""

    def __init__(self, a, b, c, data: FunctionData):
        super().__init__(data)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        return self.a * x**2 + self.b * x + self.c


class CubicFunction(Function):
    """Concrete class for cubic functions."""

    def __init__(self, a, b, c, d, data: FunctionData):
        super().__init__(data)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def evaluate(self, x):
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d


class CompositeFunction(Function):
    """Composite function f(x) = g(U(x))."""

    def __init__(self, U: Function, g: Function, data: FunctionData):
        super().__init__(data)
        self.U = U
        self.g = g

    def evaluate(self, x):
        return self.g.evaluate(self.U.evaluate(x))


class TestFunctions(unittest.TestCase):
    """Unit tests for the Function classes."""

    def test_quadratic_function_concave(self):
        """Test that a concave quadratic function is identified correctly."""
        quad_data = FunctionData(name="Concave Quadratic", domain=(-10, 10))
        quad_func = QuadraticFunction(a=-1, b=0, c=0, data=quad_data)
        self.assertTrue(quad_func.is_concave())
        self.assertFalse(quad_func.is_convex())
        # Test evaluation at specific points
        self.assertEqual(quad_func.evaluate(0), 0)
        self.assertEqual(quad_func.evaluate(1), -1)

    def test_quadratic_function_convex(self):
        """Test that a convex quadratic function is identified correctly."""
        quad_data = FunctionData(name="Convex Quadratic", domain=(-10, 10))
        quad_func = QuadraticFunction(a=1, b=0, c=0, data=quad_data)
        self.assertTrue(quad_func.is_convex())
        self.assertFalse(quad_func.is_concave())
        # Test evaluation at specific points
        self.assertEqual(quad_func.evaluate(0), 0)
        self.assertEqual(quad_func.evaluate(1), 1)

    def test_cubic_function(self):
        """Test cubic function for concavity and convexity."""
        cubic_data = FunctionData(name="Cubic Function", domain=(-2, 2))
        cubic_func = CubicFunction(a=1, b=0, c=0, d=0, data=cubic_data)
        # Cubic function x^3 is neither concave nor convex over its entire domain
        self.assertFalse(cubic_func.is_convex())
        self.assertFalse(cubic_func.is_concave())
        # Test evaluation at specific points
        self.assertEqual(cubic_func.evaluate(0), 0)
        self.assertEqual(cubic_func.evaluate(1), 1)
        self.assertEqual(cubic_func.evaluate(-1), -1)

    def test_composite_function(self):
        """Test that the composite function f(x) = g(U(x)) is concave."""
        # Define U(x) = sqrt(x)
        U_data = FunctionData(name="U(x) = sqrt(x)", domain=(0.1, 10))
        U_func = Function(U_data)
        U_func.evaluate = lambda x: np.sqrt(x)
        # Define g(z) = log(z)
        g_data = FunctionData(name="g(z) = log(z)", domain=(0.1, 10))
        g_func = Function(g_data)
        g_func.evaluate = lambda z: np.log(z)
        # Composite function f(x) = log(sqrt(x))
        comp_data = FunctionData(
            name="Composite Function f(x) = log(sqrt(x))", domain=(0.1, 10)
        )
        comp_func = CompositeFunction(U_func, g_func, data=comp_data)
        self.assertTrue(comp_func.is_concave())
        self.assertFalse(comp_func.is_convex())
        # Test evaluation at specific points
        x_values = [0.1, 1, 4, 9]
        for x in x_values:
            expected_value = 0.5 * np.log(x)
            self.assertAlmostEqual(comp_func.evaluate(x), expected_value)

    # def test_linear_function(self):
    #     """Test that a linear function is both concave and convex."""
    #     linear_data = FunctionData(name="Linear Function", domain=(-10, 10))
    #     linear_func = QuadraticFunction(a=0, b=1, c=0, data=linear_data)
    #     self.assertTrue(linear_func.is_concave())
    #     self.assertTrue(linear_func.is_convex())
    #     # Test evaluation at specific points
    #     self.assertEqual(linear_func.evaluate(0), 0)
    #     self.assertEqual(linear_func.evaluate(1), 1)
    #     self.assertEqual(linear_func.evaluate(-1), -1)

    def test_function_with_inflection_point(self):
        """Test a function with an inflection point."""
        # f(x) = x^3 - x
        cubic_data = FunctionData(
            name="Cubic Function with Inflection Point", domain=(-2, 2)
        )
        cubic_func = CubicFunction(a=1, b=0, c=-1, d=0, data=cubic_data)
        # The second derivative f''(x) = 6x
        # So f''(0) = 0 (inflection point at x=0)
        # For x < 0, f''(x) < 0 (concave)
        # For x > 0, f''(x) > 0 (convex)
        x = np.linspace(*cubic_data.domain, 1000)
        second_derivative = 6 * x
        self.assertTrue(np.any(second_derivative < 0))
        self.assertTrue(np.any(second_derivative > 0))
        # Since f'' changes sign, the function has an inflection point
        self.assertFalse(cubic_func.is_concave())
        self.assertFalse(cubic_func.is_convex())
        # Test evaluation at specific points
        self.assertEqual(cubic_func.evaluate(0), 0)
        self.assertEqual(cubic_func.evaluate(1), 0)
        self.assertEqual(cubic_func.evaluate(-1), 0)

    def test_neither_concave_nor_convex_function(self):
        """Test a function that is neither concave nor convex."""
        # f(x) = sin(x) over [0, 2π]
        sin_data = FunctionData(name="Sine Function", domain=(0, 2 * np.pi))
        sin_func = Function(sin_data)
        sin_func.evaluate = lambda x: np.sin(x)
        self.assertFalse(sin_func.is_concave())
        self.assertFalse(sin_func.is_convex())
        # Test evaluation at specific points
        x_values = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        expected_values = [0, 1, 0, -1, 0]
        for x, expected in zip(x_values, expected_values):
            self.assertAlmostEqual(sin_func.evaluate(x), expected)


if __name__ == "__main__":
    unittest.main()

    class TestFunctionProperties(unittest.TestCase):
        """Unit tests for function properties."""

        def test_linear_function(self):
            """Test that a linear function is both concave and convex."""
            linear_data = FunctionData(name="Linear Function", domain=(-10, 10))
            linear_func = QuadraticFunction(a=0, b=1, c=0, data=linear_data)
            self.assertTrue(linear_func.is_concave())
            self.assertTrue(linear_func.is_convex())
            # Test evaluation at specific points
            self.assertEqual(linear_func.evaluate(0), 0)
            self.assertEqual(linear_func.evaluate(1), 1)
            self.assertEqual(linear_func.evaluate(-1), -1)

        def test_function_with_inflection_point(self):
            """Test a function with an inflection point."""
            # f(x) = x^3 - x
            cubic_data = FunctionData(
                name="Cubic Function with Inflection Point", domain=(-2, 2)
            )
            cubic_func = CubicFunction(a=1, b=0, c=-1, d=0, data=cubic_data)
            # The second derivative f''(x) = 6x
            # So f''(0) = 0 (inflection point at x=0)
            # For x < 0, f''(x) < 0 (concave)
            # For x > 0, f''(x) > 0 (convex)
            x = np.linspace(*cubic_data.domain, 1000)
            second_derivative = 6 * x
            self.assertTrue(np.any(second_derivative < 0))
            self.assertTrue(np.any(second_derivative > 0))
            # Since f'' changes sign, the function has an inflection point
            self.assertFalse(cubic_func.is_concave())
            self.assertFalse(cubic_func.is_convex())
            # Test evaluation at specific points
            self.assertEqual(cubic_func.evaluate(0), 0)
            self.assertEqual(cubic_func.evaluate(1), 0)
            self.assertEqual(cubic_func.evaluate(-1), 0)

        def test_neither_concave_nor_convex_function(self):
            """Test a function that is neither concave nor convex."""
            # f(x) = sin(x) over [0, 2π]
            sin_data = FunctionData(name="Sine Function", domain=(0, 2 * np.pi))
            sin_func = Function(sin_data)
            sin_func.evaluate = lambda x: np.sin(x)
            self.assertFalse(sin_func.is_concave())
            self.assertFalse(sin_func.is_convex())
            # Test evaluation at specific points
            x_values = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
            expected_values = [0, 1, 0, -1, 0]
            for x, expected in zip(x_values, expected_values):
                self.assertAlmostEqual(sin_func.evaluate(x), expected)

    if __name__ == "__main__":
        unittest.main()
