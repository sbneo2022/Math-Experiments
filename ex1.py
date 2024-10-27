from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FunctionData:
    """Data class to store function details."""

    name: str
    domain: tuple
    is_concave: bool = False
    is_convex: bool = False


class Function(ABC):
    """Abstract base class for mathematical functions."""

    def __init__(self, data: FunctionData):
        self.data = data

    @abstractmethod
    def evaluate(self, x) -> float:
        """Evaluate the function at a given point."""
        pass

    def is_concave(self) -> bool:
        """Check if the function is concave."""
        x = np.linspace(*self.data.domain, 1000)
        second_derivative = np.gradient(np.gradient(self.evaluate(x)))
        return np.all(second_derivative <= 0)

    def is_convex(self) -> bool:
        """Check if the function is convex."""
        x = np.linspace(*self.data.domain, 1000)
        second_derivative = np.gradient(np.gradient(self.evaluate(x)))
        return np.all(second_derivative >= 0)

    def plot(self):
        """Plot the function."""
        x = np.linspace(*self.data.domain, 1000)
        y = self.evaluate(x)
        plt.plot(x, y, label=self.data.name)
        plt.title(f"{self.data.name} in the range {self.data.domain}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()


class QuadraticFunction(Function):
    """Quadratic function f(x) = ax^2 + bx + c."""

    def __init__(self, data: FunctionData, a: float, b: float, c: float):
        super().__init__(data)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x) -> float:
        return self.a * x**2 + self.b * x + self.c


class CubicFunction(Function):
    """Cubic function f(x) = ax^3 + bx^2 + cx + d."""

    def __init__(self, data: FunctionData, a: float, b: float, c: float, d: float):
        super().__init__(data)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def evaluate(self, x) -> float:
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d


class LogFunction(Function):
    """Logarithmic function f(x) = log(x)."""

    def __init__(self, data: FunctionData):
        super().__init__(data)

    def evaluate(self, x) -> float:
        return np.log(x)


class CompositeFunction(Function):
    """Composite function g(U(x))."""

    def __init__(self, U: Function, g: Function, data: FunctionData):
        super().__init__(data)
        self.U = U
        self.g = g

    def evaluate(self, x):
        return self.g.evaluate(self.U.evaluate(x))


# Example usage
def main():
    # Define a concave quadratic function
    quad_data = FunctionData(name="Concave Quadratic", domain=(-10, 10))
    quad_func = QuadraticFunction(a=-1, b=0, c=0, data=quad_data)
    print(f"{quad_func.data.name} is concave: {quad_func.is_concave()}")
    quad_func.plot()

    # Define a convex quadratic function
    quad_data_convex = FunctionData(name="Convex Quadratic", domain=(-10, 10))
    quad_func_convex = QuadraticFunction(a=1, b=0, c=0, data=quad_data_convex)
    print(f"{quad_func_convex.data.name} is convex: {quad_func_convex.is_convex()}")
    quad_func_convex.plot()

    # Define a cubic function with an inflection point
    cubic_data = FunctionData(name="Cubic Function", domain=(-2, 2))
    cubic_func = CubicFunction(a=1, b=0, c=0, d=0, data=cubic_data)
    print(f"{cubic_func.data.name} is convex: {cubic_func.is_convex()}")
    print(f"{cubic_func.data.name} is concave: {cubic_func.is_concave()}")
    cubic_func.plot()

    # Define a composite function f(x) = g(U(x))
    U_data = FunctionData(name="U(x) = sqrt(x)", domain=(0.1, 10))
    U_func = Function(U_data)
    U_func.evaluate = lambda x: np.sqrt(x)  # Define U(x)

    g_data = FunctionData(name="g(z) = log(z)", domain=(0.1, 10))
    g_func = Function(g_data)
    g_func.evaluate = lambda z: np.log(z)  # Define g(z)

    comp_data = FunctionData(
        name="Composite Function f(x) = log(sqrt(x))", domain=(0.1, 10)
    )
    comp_func = CompositeFunction(U_func, g_func, data=comp_data)
    print(f"{comp_func.data.name} is concave: {comp_func.is_concave()}")
    comp_func.plot()


if __name__ == "__main__":
    main()
