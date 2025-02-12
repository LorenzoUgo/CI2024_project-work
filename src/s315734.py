import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return np.divide(np.multiply(np.sinh(12.781375081560109), x[1]), -0.2938385126521815)


def f3(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.absolute(x[1]), np.reciprocal(x[0]))

def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.arctanh(4.544818841950184), np.absolute(x[1]))

def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(x[0], np.floor_divide(x[0], 0.3189202384321422))

def f6(x: np.ndarray) -> np.ndarray:
    return np.add(x[1], np.arccosh(x[1]))

def f7(x: np.ndarray) -> np.ndarray:
    return np.exp((np.add(np.multiply(x[1], x[1]), np.arctan(np.log2(np.add(x[0], x[0]))))))

def f8(x: np.ndarray) -> np.ndarray:
    return np.sinh(np.arccosh(x[0]))
