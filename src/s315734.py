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
    return np.add(np.add(-1.9719166033534583, x[0]), np.exp2(4.2361333635788725))

def f4(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.divide(x[0], -5.295734315518481), np.tanh(x[0]))

def f5(x: np.ndarray) -> np.ndarray:
    return np.remainder(np.log10(x[1]), x[0])

def f6(x: np.ndarray) -> np.ndarray:
    #(x0 add (arctan x0))
    return np.add(np.cosh(x[1]), np.reciprocal(x[0]))

def f7(x: np.ndarray) -> np.ndarray:
    return np.add(np.exp(x[0]), x[0])
                  
def f8(x: np.ndarray) -> np.ndarray:
    return np.exp(np.floor_divide(x[0], 0.1709665900941516))
