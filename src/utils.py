##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
import itertools
from typing import Callable
import numbers
import warnings
import random
import copy
from collections import deque
from src.draw import draw
import matplotlib.pyplot as plt
import types
from functools import total_ordering
from tqdm import tqdm
from copy import deepcopy

import src.s315734 as s315734
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)


numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.expm1, np.exp2, np.log, np.log1p, np.log10, np.log2],
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin, np.hypot],
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.square, np.absolute, np.reciprocal],
    #"rounding":         [np.ceil, np.round, np.trunc],           #SOLO come supporto post-generazione per ridurre le cifre decimali dei valori
}

numpy_cost = {np.e, np.pi, np.euler_gamma, np.finfo(float).eps}

@total_ordering
class Individual:
    def __init__(self, f: 'Node', x: np.ndarray[float] = None, y: np.ndarray[float] = None):
        self.SymRegTree = f
        if x is not None and y is not None:
            self.MSE = self.__compute_MSE__(x, y)
            self.fitness = self.__compute_fitness__()

    def __eq__(self, other: 'Individual'):
        if not isinstance(other, Individual):
            return False
        
        return self.SymRegTree == other.SymRegTree
    
    def __lt__(self, other: 'Individual'):
        if isinstance(other, Individual):
            return self.fitness < other.fitness
    
        return NotImplemented
    
    def __compute_MSE__(self, x: np.ndarray[float], y: np.ndarray[float]):
        MSE = 100*np.square(y - self.deploy_function(x)).sum()/len(y)
        if np.isnan(MSE): 
            return np.inf
        return MSE
    
    def __compute_fitness__(self):
        ''' MSE + f_length '''
        return self.MSE
    
    def compute_metrics(self, x: np.ndarray[float], y: np.ndarray[float]):
        self.MSE = self.__compute_MSE__(x, y)
        self.fitness = self.__compute_fitness__()
        
    def show_function(self):
        ##  print(self.SymRegTree._name)
        ##  if len(self.SymRegTree._successor)>1:
        ##      print(self.SymRegTree._successor[1]._name)
        ##  print(self.SymRegTree._successor[0]._name)

        print(self.SymRegTree)

    def deploy_function(self, val):
        ## self.show_function()
        return self.SymRegTree.apply(val)

    def gene_mutation(self, vars):
        ''' 
        What do I want to mutate? 
        Can I mutate a node into a different type?
        '''
        type_mu = random.choice(range(3))
        
        if type_mu==0:
            ## NEW Operand
            self.SymRegTree.mutate(random.choice(list(itertools.chain(*numpy_funct.values()))))
        if type_mu==1:
            ## Change in the Value
            self.SymRegTree.mutate(random.gauss(mu=0, sigma=1))
        elif type_mu==2 and len(vars) > 1:
            ## NEW Variable, if multiple variable
            self.SymRegTree.mutate(random.choice(vars))

    def get_fitness(self):
        return self.fitness
    
    def get_MSE(self):
        return self.MSE

    def show_results(self):
        print(f"At the end, the current individual is the best:")
        print(f"\t--> Fitenss = {self.fitness}\n\t--> MSE = {self.MSE}")
        print(f"\t--> Function: {self.SymRegTree}")

class Node:
    _value: int|str|Callable
    _name: str
    _successor: tuple['Node', ...]

    def __init__(self, node=None, successors=None):

        if callable(node):
            ''' The node is a function --> Must have some successors --> Not a leaf '''
            def _f(*_args, **_kwargs):
                try:
                    f = node(*_args)
                    return f
                except RuntimeWarning as e:
                    ##  print(e, self._successor) 
                    return np.nan
                

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

    def __eq__(self, other: "Node"):
        if not isinstance(other, Node):
            return False
        
        if self._name != other._name:
            return False
    
        if len(self._successor) != len(other._successor):
            return False

        return all(s1 == s2 for s1, s2 in zip(self._successor, other._successor))

    def __apply_f__(self, var_value):
        if self._leaf:
            if isinstance(self._value, str):
                int_ = int(self._value[1:])
                return var_value[int_]
            else:
                return self._value
        
        return self._value(*[next.__apply_f__(var_value) for next in self._successor])
    
    def __is_equivalent(self, f1, f2) -> bool:
        if isinstance(f1, int) and isinstance(f2, str):
            return type(f1) == type(f1)
        else:
            return isinstance(f1, types.FunctionType) and isinstance(f2, np.ufunc)
    
    def mutate(self, new):
        if self.__is_equivalent(self._value, new) and random.random() < 0.5:   ##  TODO update...
            self.__mutate__(new)
            return True

        for child in self._successor:
            if child.mutate(new):
                return True
        
        return False

    def __mutate__(self, new_val: np.ufunc):
        if callable(new_val):
            ##  TODO problem to face: can change ufunct.nin --> need to change the number of successors
            def _f(*_args, **_kwargs):
                try:
                    f = new_val(*_args)
                    return f
                except RuntimeWarning as e:
                    ##  print(e, self._successor) 
                    return np.nan
                ##  return new_val(*_args)
            self._value = _f
            self._name = new_val.__name__

            if new_val.nin > len(self._successor):
                if isinstance(self._successor[0]._value, str):
                    self._successor = (self._successor[0], Node(random.gauss(mu=0, sigma=1)))
                else:
                    ## How to insert a different variable ?
                    self._successor = (self._successor[0], Node("x0"))

            elif new_val.nin < len(self._successor):
                self._successor = (random.choice(self._successor),)
        
            elif new_val.nin == len(self._successor):
                self._successor = tuple(random.sample(self._successor, len(self._successor)))
                
        elif isinstance(new_val, str):
            self._value = new_val
            self._name = new_val
        
        elif isinstance(new_val, numbers.Number):
            self._value += new_val
            self._name = str(self._value)
            
    def apply(self, var_value):
        return self.__apply_f__(var_value)
    
    def add_successor(self, n:'Node'):
        if not self._leaf:
            #if len(self._successor) == self._value.nin :
            #    for node in list(self._successor):
            #        node.add_successor(n)
            #else:
                self._successor = tuple(list(self._successor)+[n])
                #return
        else:
            return
          
    def get_level(self):
        if self._leaf:
            return 0
        
        return 1 + max(child.get_level() for child in self._successor)

    '''def draw(self):
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None'''


class Genetic_Algorithm:
    _population_size: int
    _num_offsprings: int
    _num_islands: int
    _population: list[Individual]
    _num_generations: int
    _num_eras: int
    _variables: int

    def __init__(self, pop_size, off_num, num_gen, num_eras, num_var, num_isl=1):
        self._num_islands = num_isl
        self._num_offsprings = off_num
        self._population_size = pop_size
        self._num_generations = num_gen
        self._num_eras = num_eras
        self._variables = [self.formatting(i) for i in range(num_var)]
        ## self._population = {i: self.__random_init__() for i in range(self._num_islands)}
        self._population = self.__random_init__()

    def formatting(self, idx: int) -> str:
        return f"x{idx}"

    def __random_init__(self):
        return [Individual(self.__random_tree__(len(self._variables))) for _ in range(self._population_size)]
    
    '''def __random_init__(self):
        population = []

        for i in range(self._population_size):
            if len(self._variables) == 1:
                f = Node(f"x0")
                ind = Individual(f)
                population.append(ind)
            else:
                j = 0
                operands = []
                while j < range(int(np.ceil(len(self._variables)*1.5))):
                    op = random.choice(list(itertools.chain(*numpy_funct.values())))
                    operands.append(op)
                    if op.nin > 1:
                        j += 1

                var = 0
                op = random.choice(operands)
                f = Node(op)
                operands.remove(op)

                while len(operands):
                    if random.random() < 0.5:
                        op = random.choice(operands)
                        f.add_successor(Node(op))
                        operands.remove(op)
                    else:
                        if random.random() < 0.5:
                            f.add_successor(Node(random.uniform(-5, 5)))
                        else:
                            # Fare random anche quale variabile viene inserita
                            f.add_successor(Node(f"x{var}"))
                            var += 1

        return population'''
    
    def __random_tree__(self, max_depth: int, current_depth: int = 0):
        """Return a random tree generated according to num_variables"""
        # Return a leaf according to a probability that depends on how deep we are
        if random.random() < (current_depth / max_depth):
            return self.__create_leaf__()
        
        # Create a function node
        op = random.choice(list(itertools.chain(*numpy_funct.values())))
        successors = [self.__random_tree__(max_depth, current_depth + 1) for _ in range(op.nin)]

        return Node(op, successors)
    
    def __create_leaf__(self) -> Node:
        """Create a leaf, either a const or a variable"""
        if random.random() < 0.3:
            return Node(random.uniform(-1, 1))
        else:
            return Node(random.choice(self._variables))
    
    def __crossover__(self, ind1: Individual, ind2: Individual) -> Individual:
        ... ## TODO

    def __random_mutation__(self, ind: Individual) -> Individual:
        ##  Can I random mutate val into var into funct and viceversa ??  ##
        new_ind = deepcopy(ind)
        new_ind.gene_mutation(self._variables)

        return new_ind

    def __parent_selection__(self)-> tuple[Individual, Individual]:
        p1 = random.choice(self._population)
        p2 = random.choice(self._population)
        while p1 == p2:
            p1 = random.choice(self._population)
            p2 = random.choice(self._population)
        return p1, p2

    def __selection__(self)-> Individual:
        return random.choice(self._population)

    def __survival__(self, offsprings: list[Individual]) -> list[Individual]:
        extended_population = self._population + offsprings
        extended_population.sort(key=lambda ind: ind.get_fitness())     # ORDERING FROM BEST TO WORSE
        print([ind.SymRegTree._name for ind in extended_population])
        self._population = extended_population[:self._population_size]  # SURVIVAL SELECTION

    def variable_checking(self, value):
        if value.shape[0] != len(self._variables):
            raise ValueError(f"This problem require {len(self._variables)} variables, but you passed only {value.shape[0]} variables !")
    
    def show_population(self):
        for ind in self._population:
            ind.show_function()

    def show_individual(self):
        self._population[0].show_function()

    def deploy_population(self, val):
        for ind in self._population:
            print("Computed value: ", ind.deploy_function(val))

    def start(self, x: np.ndarray[float], y: np.ndarray[float]):

        for ind in self._population:
            ind.compute_metrics(x, y)

        best_ind_history = list()

        for g in tqdm(range(self._num_generations), desc="Generation", leave=True):
            offsprings = list()
            for o in tqdm(range(self._num_offsprings), desc="Offspring generated", leave=False):
                if random.random() > 2.0:
                    ind1, ind2 = self.__parent_selection__(self._population)    ## Usare la tecnica dell'UNPACKING
                    ind = self.__crossover__(ind1, ind2)
                else:
                    ind = self.__selection__()

                off = self.__random_mutation__(ind)
                del ind
            
                off.compute_metrics(x, y)
                offsprings.append(off)
            
            self.__survival__(offsprings)
            best_ind_history.append(deepcopy(self._population[0]))

        self._population[0].show_results()
        return best_ind_history

    def __save_best_ind__(self, history: list):
        history.append(self._population.sort())
        return history
    
    def plot_fitness_history(self, history: list[Individual]):
        generations = list(range(1, self._num_generations+1))
        fitness_history = [i.get_fitness() for i in history]
        plt.plot(generations, fitness_history, marker='o', linestyle='-', color='r')
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.show()

    def plot_MSE_history(self, history: list[Individual]):
        generations = list(range(1, self._num_generations))
        fitness_history = [i.get_MSE() for i in history]
        plt.plot(generations, fitness_history, marker='o', linestyle='-', color='r')
        plt.title('MSE Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

### - - - - - ###
def function_cod(num: int) -> list[str]:
    return [getattr(s315734, f"f{i}") for i in range(num+1)]

def data_loader(problem_num: int) -> tuple[np.ndarray, np.ndarray]:
    problem = np.load(f'data/problem_{problem_num}.npz')
    x = problem['x']
    y = problem['y']
    print(f"Problem type {problem_num}: ", x.shape, y.shape)
    print(f"Number of variables: ", x.shape[0])
    
    return x, y

def data_split(x: np.ndarray, y: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    '''
        Dataset split: 70% train and 30% test.
    '''
    idx_split = int(x.shape[1]*70/100)

    _train = (x[:, :idx_split], y[:idx_split])
    _test = (x[:, idx_split:], y[idx_split:])

    print(f"Train dataset: ", _train[0].shape, _train[1].shape)
    print(f"Test dataset: ", _test[0].shape, _test[1].shape)

    return _train, _test

def test(ind: Individual, x: list, y:list) -> float:
    MSE = 100*np.square(y - ind.deploy_function(x)).sum()/len(y)
    print(f"MSE (real) : {MSE:g}")
    return MSE

def train(ind: Individual, x: list, y:list) -> float:
    MSE = 100*np.square(y - ind.deploy_function(x)).sum()/len(y)
    print(f"MSE (train) : {MSE:g}")
    return MSE

###
  # L'ordinamento conta nelle operazioni binarie --> Mettere in atto controlli/randomicità
  # Capire se una fuzione e uni o bi arg è importante quando si fanno modifiche o tagli !!
  # Longer function --> Penalty
  # Genetic Algo / Evolutionary Strategy / ... 
  # Graph representation of function
  # ###