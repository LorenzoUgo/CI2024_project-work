##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
from typing import Callable
import numbers
import warnings
from draw import draw

numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.expm1, np.exp2, np.log, np.log1p, np.log10, np.log2],
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin, np.hypot],
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.square, np.absolute, np.clip, np. reciprocal],
    "rounding":         [np.ceil, np.round, np.trunc],           #SOLO come supporto post-generazione per ridurre le cifre decimali dei valori
}

class Individual:
    def __init__(self, function: 'Node'):
        self.MSE = self.__MSE__
        self.function = function
        self.fitness = self.__fitness__()

    def __MSE__(self, x, y):
        return 100*np.square(y - deploy_function(self.function)(x)).sum()/len(y)
    
    def __fitness__(self):
        ''' MSE + f_length '''
        ...


class Node:
    _value: int
    #_func: function
    _name: str
    _successor: tuple['Node']

    def __init__(self, node=None, successors=None):
        if callable(node):
            ''' The node is a function --> Must have some successors --> Not a leaf '''
            def _f(*_args, **_kwargs):
                return node(*_args)

            self._func = _f
            self._successor = tuple(successors)
            self._leaf = False
            self._name = node.__name__

        elif isinstance(node, numbers.Number):
            ''' The node is a number --> Must be a leaf '''
            self._value = node
            self._successor = tuple()
            self._leaf = True
            self._name = str(node)

        elif isinstance(node, str):
            ''' The node is a variable --> Must be a leaf '''
            self._value = node
            self._successor = tuple()
            self._leaf = True
            self._name = node

    def __str__(self):
        num_child = sum(x is not None for x in self._successor)
        if num_child == 2:
            return f"({self._successor[0]} {self._name} {self._successor[1]})"
        elif num_child == 1:
            return f"({self._name} {self._successor[0] if self._successor[0] is not None else self._successor[1]})"
        
        return self._name

    def draw(self):
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None

    def __apply_f__(self, var_value):
        if not len(self._successor):
            if isinstance(self._value, str):
                return var_value
            else:
                return self._value
        
        return self._func(*[next.__apply_f__(var_value) for next in self._successor])


def function_cod(num: int) -> list[str]:
    return [f"f{i}" for i in range(num)]

def deploy_function(f: list):
    def funct(x):
        ... 
    return funct

def data_split(problem_num: int):
    '''
        Dataset split: 70% train and 30% test.
    '''
    data = np.load(f'../data/problem_{problem_num}.npz')
    x = data['x']
    y = data['y']
    idx_split = x.shape[1]*70/100

    return (x[:, :idx_split], y[:idx_split]), (x[:, idx_split:], y[idx_split:])

def test(function: list, x: list, y:list) -> float:
    MSE = 100*np.square(y - deploy_function(function)(x)).sum()/len(y)
    print(f"MSE (real) : {MSE:g}")
    return MSE

def train(function: list, x: list, y:list) -> float:
    MSE = 100*np.square(y - deploy_function(function)(x)).sum()/len(y)
    print(f"MSE (train) : {MSE:g}")
    return MSE

###
  # Longer function --> Penalty
  # Genetic Algo / Evolutionary Strategy / ... 
  # Graph representation of function
  # ###