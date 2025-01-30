##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
import itertools
from typing import Callable
import numbers
import warnings
import random
from draw import draw

numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.expm1, np.exp2, np.log, np.log1p, np.log10, np.log2],
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin, np.hypot],
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.square, np.absolute, np.clip, np. reciprocal],
    "rounding":         [np.ceil, np.round, np.trunc],           #SOLO come supporto post-generazione per ridurre le cifre decimali dei valori
}

numpy_cost = {np.e, np.pi, np.euler_gamma, np.finfo(float).eps}

class Individual:
    def __init__(self, f: 'Node'):
        self.MSE = self.__MSE__
        self.SymRegTree = f
        self.fitness = self.__fitness__()

    def __MSE__(self, x, y):
        return 100*np.square(y - deploy_function(self.function)(x)).sum()/len(y)
    
    def __fitness__(self):
        ''' MSE + f_length '''
        ...

    def gene_mutation(self, new):
        ''' What do I want to mutate? '''
        if callable(new):
            self.SymRegTree.replace_function(new)
        elif isinstance(new, numbers.Number):
            self.SymRegTree.replace_const(new)
        elif isinstance(new, str):
            self.SymRegTree.replace_var(new)

class Node:
    _value: int
    _name: str
    _successor: tuple['Node']

    def __init__(self, node=None, successors=None):

        if callable(node):
            ''' The node is a function --> Must have some successors --> Not a leaf '''
            def _f(*_args, **_kwargs):
                return node(*_args)

            self._value = _f
            self._leaf = False
            self._name = node.__name__

        elif isinstance(node, numbers.Number):
            ''' The node is a number --> A leaf '''
            self._value = node
            self._leaf = True
            self._name = str(node)

        elif isinstance(node, str):
            ''' The node is a variable --> A leaf '''
            self._value = node
            self._leaf = True
            self._name = node
        
        self._successor = tuple()
        if successors is not None:
            if not isinstance(successors, list):
                successors = list(successors)
            for item in list(successors):
                self.add_successor(item)
        
    def __str__(self):
        num_child = sum(x is not None for x in self._successor)
        if num_child == 2:
            return f"({self._successor[0]} {self._name} {self._successor[1]})"
        elif num_child == 1:
            return f"({self._name} {self._successor[0] if self._successor[0] is not None else self._successor[1]})"
        
        return self._name

    def __apply_f__(self, var_value):
        if self._leaf:
            if isinstance(self._value, str):
                return var_value
            else:
                return self._value
        
        return self._value(*[next.__apply_f__(var_value) for next in self._successor])
    
    def apply(self, var_value):
        return self.__apply_f__(var_value)
    
    def add_successor(self, n:'Node'):
        if len(self._successor) == 2 :
            raise IndexError("Can't have more than 2 successor")
        
        self._successor = tuple(list(self._successor).append(n))
    
    def get_level(self):
        if self._leaf:
            return 0
        
        return 1 + max(child.get_level() for child in self._successor)

    def replace_function(self, new_f):
        if not (0 <= index < len(self._successor)):
            raise IndexError("Index out of range for successor tuple")
        
        new_successors = tuple(
            new_node if i == index else child for i, child in enumerate(self._successor)
        )
        return 

    def replace_const(self, new_c):
        if not (0 <= index < len(self._successor)):
            raise IndexError("Index out of range for successor tuple")
        
        new_successors = tuple(
            new_node if i == index else child for i, child in enumerate(self._successor)
        )
        return 

    def replace_var(self, new_v):
        if not (0 <= index < len(self._successor)):
            raise IndexError("Index out of range for successor tuple")
        
        new_successors = tuple(
            new_node if i == index else child for i, child in enumerate(self._successor)
        )
        return 

    def draw(self):
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None


class Genetic_Algorithm:
    _population_size: int
    _offsprings: int
    _num_islands: int
    _population: list[Individual]
    _num_generations: int
    _num_eras: int
    _num_variables: int

    def __init__(self, pop_size, off_num, num_isl, num_gen, num_eras, num_var):
        self._num_islands = num_isl
        self._offsprings = off_num
        self._population_size = pop_size
        self._num_generations = num_gen
        self._num_eras = num_eras
        self._num_variables = num_var
        self._population = self.__random_init__()


    def __random_init__(self):
        population = []

        var_list = [f"x{i}" for i in range(self._num_variables)]

        for i in range(self._population_size):
            if self._num_variables == 1:
                f = Node(var_list[0])
                ind = Individual(f)
                population.append(ind)
            else:
                j = 0
                operand = []
                while j < range(self._num_variables-1):
                    op = random.choice(list(itertools.chain(*numpy_funct.values())))
                    operand.append(op)
                    if op.nin > 1:
                        j += 1

                
        return population




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
  # Capire se una fuzione e uni o bi arg è importante quando si fanno modifiche o tagli !!
  # Longer function --> Penalty
  # Genetic Algo / Evolutionary Strategy / ... 
  # Graph representation of function
  # ###