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
from collections import deque, defaultdict
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

        ##  ADD with crossover computation: self.__compute_level_dict__()

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
    
    def __select_random_subtree__(self, avoid_root: bool = False, size_bias: bool = False) -> "Node":
        nodes = self.SymRegTree.get_all_nodes()

        if avoid_root and len(nodes) > 1:
            nodes.remove(self.SymRegTree)  # Evita la radice

        if size_bias:
            depths = [node.get_depth() for node in nodes]
            
            # Calcoliamo il peso: più probabilità ai nodi di profondità intermedia
            weights = [abs(random.gauss(mu=d, sigma=1)) for d in depths]
            total = sum(weights)
            weights = [w / total for w in weights] 
            return random.choices(nodes, weights=weights, k=1)[0]  if nodes else None
        
        return random.choice(nodes) if nodes else None

    def __generate_random_subtree__(self, vars: list[str], max_depth: int, current_depth: int = 0, island: str = "unique"):
        """Return a random tree generated according to num_variables"""
        # Return a leaf according to a probability that depends on how deep we are
        if random.random() < (current_depth / max_depth):
            if random.random() < 0.3:
                return Node(random.uniform(-1, 1))
            else:
                return Node(random.choice(vars))
        
        # Create a function node
        if island == "unique":
            op = random.choice(list(itertools.chain(*numpy_funct.values())))
        else:
            op = random.choice(numpy_funct[island])

        successors = [self.__generate_random_subtree__(vars, max_depth, current_depth + 1, island) for _ in range(op.nin)]

        return Node(op, successors)

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

    def leaf_mutation(self, vars: list[str]):
        ''''''
        node = self.SymRegTree.get_random_node(type = "leaf")

        ## Change DIRECTLY the node OR pass to the tree the trget node and search and reach it
        if isinstance(node._value, numbers.Number):
            ## Change in the Value
            node.mutate(random.gauss(mu=0, sigma=1))
        elif isinstance(node._value, str):
            ## NEW Variable
            node.mutate(random.choice(vars), len(vars))

    def structural_mutation(self):
        node = self.SymRegTree.get_random_node(type = "node")   ##  Nodo target da cercare 

        if random.choice(["insert_node", "remove_node"]) == "insert_node":
            return self.SymRegTree.insert_intermediate_node(node)
        else:
            return self.SymRegTree.remove_and_merge(node)

    def function_mutation(self, vars:list[str], mutation_rate: float, island: str = "unique"):
        node = self.SymRegTree.get_random_node(type = "node")

        if mutation_rate > 0.3:
            w = [0.3, 0.7]
        else:
            w = [0.7, 0.3]
        
        if random.choices(["substitution", "new_subtree"], weights=w)[0] == "substitution":
            ## NEW Operand
            if island == "unique":
                node.mutate(random.choice(list(itertools.chain(*numpy_funct.values()))), len(vars))
            else:
                node.mutate(random.choice(numpy_funct[island]), len(vars)) 
            
        else:
            new_subtree = self.__generate_random_subtree__(vars, max_depth=node.get_depth(), island=island)
            self.SymRegTree.replace_subtree(node, new_subtree)
        
    def gene_crossover(self, avoid_root: bool= False, size_bias:bool = False):
        return self.__select_random_subtree__(avoid_root, size_bias), deepcopy(self.SymRegTree)

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
    
    def __insert__(self, target: "Node", island: str = "unique"):

        if island == "unique":
            op = random.choice(list(itertools.chain(*numpy_funct.values())))
        else:
            op = random.choice(numpy_funct[island])

        if op.nin == 1:
            if len(target._successor) == 1:
                node = Node(op, target._successor)
                target._successor = (node, )

            else:
                if random.random() < 0.5:
                    node = Node(op, target._successor[0])
                    target._successor = (node, target._successor[1])
                
                else:
                    node = Node(op, target._successor[1])
                    target._successor = (target._successor[0], node)
            
        else:
            if len(target._successor) == 1:
                node = Node(op, [target._successor[0], Node(random.uniform(-1, 1), ())])
                target._successor = (node, )

            else:
                if random.random() < 0.5:
                    node = Node(op, [target._successor[0], Node(random.uniform(-1, 1), ())])
                    target._successor = (node, target._successor[1])
                
                else:
                    node = Node(op, [Node(random.uniform(-1, 1), ()), target._successor[1]])
                    target._successor = (target._successor[0], node)

    def __remove__(self, target: "Node"):
        if not self._successor or not target:
            return  # Avoid Errors
        
        idx = self._successor.index(target)
        
        if len(target._successor) == 1:
            if len(self._successor) == 1:
                self._successor = (target._successor[0], )

            else:
                if idx == 0:
                    self._successor = (target._successor[0], self._successor[1])

                else:
                    self._successor = (self._successor[0], target._successor[0])

        else:
            if len(self._successor) == 1:
                if random.random() < 0.5:    
                    self._successor = (target._successor[0], )
                else:
                    self._successor = (target._successor[1], )

            else:
                if idx == 0:
                    if random.random() < 0.5:
                        self._successor = (target._successor[1], self._successor[1])
                    else:
                        self._successor = (target._successor[0], self._successor[1])

                else:
                    if random.random() < 0.5:
                        self._successor = (self._successor[0], target._successor[0])
                    else:
                        self._successor = (self._successor[0], target._successor[1])
                
        del target
                
    def get_all_nodes(self, type: str = "all") -> list["Node"]:
        """ Ritorna tutti i nodi dell'albero """
        nodes = []

        def traverse(node: Node):
            if node:
                if type == "all":
                    nodes.append(node)

                elif type == "leaf":
                    if not node._successor:     ##  If it has not successors, it's a leaf
                        nodes.append(node)

                elif type == "node":
                    if node._successor:     ##  If it has successors, it's a node
                        nodes.append(node) 

                for child in node._successor:
                    traverse(child)

        traverse(self)
        return nodes

    def get_random_node(self, type:str = "all") -> "Node":

        return random.choice(self.get_all_nodes(type))

    def insert_intermediate_node(self, target: "Node", island: str = "unique"):
        if self.__is_equivalent(self._value, target):
            self.__insert__(target, island=island)
            return self
        else:
            for child in self._successor:
                child.insert_intermediate_node(target, island)
        
    def remove_and_merge(self, target):
        if self._successor:
            if target in self._successor:
                self.__remove__(target)
                return self
            else:
                for child in self._successor:
                    child.remove_and_merge(target)

    def mutate(self, new, num_var: int = 1):
        self.__mutate__(new, num_var)

    def __mutate__(self, new_val: np.ufunc, num_var: int = 1):
        if callable(new_val):
            ##  TODO ...
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
                    self._successor = (self._successor[0], Node(f"x{int(random.randrange(num_var))}"))
                    
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
    
    def replace_subtree(self, target, replacement):
        if self == target:
            return replacement
        new_successors = [child.replace_subtree(target, replacement) for child in self._successor]
        if callable(self._value):
            return Node(getattr(np, self._name), new_successors)
        else:
            return Node(self._value, new_successors)
    
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
          
    def get_depth(self):
        """Calcola la profondità del nodo corrente rispetto alla radice."""
        if not self._successor:
            return 0
        return 1 + max(child.get_depth() for child in self._successor)

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
    _populations: dict[str: list[Individual]]
    ##_population: list[Individual]
    _num_generations: int
    _num_eras: int
    _variables: int

    def __init__(self, pop_size, off_num, num_gen, num_eras, num_var, num_isl=len(numpy_funct)):
        self._num_islands = num_isl
        self._num_offsprings = off_num
        self._population_size = pop_size
        self._num_generations = num_gen
        self._num_eras = num_eras
        self._variables = [self.formatting(i) for i in range(num_var)]
        self._mutation_rate = 0.3       ##  High Exploration

        if self._num_islands == 1:
            self._populations = {"unique": self.__random_init_island__()}
        else:
            self._populations = {i: self.__random_init_island__(i) for i in numpy_funct}

    def formatting(self, idx: int) -> str:
        return f"x{idx}"

    def __random_init_island__(self, island: str = "unique") -> list[Individual]:
        return [Individual(self.__random_tree__(island=island, max_depth=len(self._variables))) for _ in range(self._population_size)]
    
    def __random_tree__(self, max_depth: int, current_depth: int = 0, island: str="unique") -> Node:
        """Return a random tree generated according to num_variables"""
        # Return a leaf according to a probability that depends on how deep we are
        if random.random() < (current_depth / max_depth):
            return self.__create_leaf__()
        
        # Create a function node
        if island == "unique":
            op = random.choice(list(itertools.chain(*numpy_funct.values())))
        else:
            op = random.choice(numpy_funct[island])
        successors = [self.__random_tree__(max_depth, current_depth + 1, island) for _ in range(op.nin)]

        return Node(op, successors)
    
    def __create_leaf__(self) -> Node:
        """Create a leaf, either a const or a variable"""
        if random.random() < 0.3:
            return Node(random.uniform(-1, 1))
        else:
            return Node(random.choice(self._variables))
    
    def __crossover__(self, ind1: Individual, ind2: Individual) -> tuple[Individual, Individual]:
        if not isinstance(ind1, Individual) or not isinstance(ind2, Individual):
            raise ValueError("Crossover is possible only between two individuals.")
        
        subtree1, tree1 = ind1.gene_crossover(avoid_root=True, size_bias=True)
        subtree2, tree2 = ind2.gene_crossover(avoid_root=True, size_bias=True)

        if subtree1 is None or subtree2 is None:
            return ind1, ind2   ## No crossover is possible
        
        new_tree1 = tree1.replace_subtree(subtree1, subtree2)
        new_tree2 = tree2.replace_subtree(subtree2, subtree1)

        return Individual(new_tree1), Individual(new_tree2)

    def __mutation__(self, ind: Individual, island: str = "unique") -> Individual: 
        if random.random() > self._mutation_rate :
            return # No mutation --> return ind
        
        ## ITs it necessary? --> new_ind = deepcopy(ind)
        ##  Can I random mutate val into var into funct and viceversa ??  ##

        w = [0.4, 0.3, 0.3] if self._mutation_rate > 0.3 else [0.5, 0.3, 0.2]    # Mutazione var, val, funct / Mutazione subtree / Mutazione strutturale
        
        mutation_type = random.choices(["leaf", "function", "structural"], weights=w)[0]
        
        if mutation_type == "structural":
            ind.structural_mutation()
            #   return new_ind.structural_mutation()
        elif mutation_type == "leaf":
            ind.leaf_mutation(self._variables)
            #   return new_ind.leaf_mutation(self._variables)
        elif mutation_type == "function":
            ind.function_mutation(self._variables, island=island)
            #   return new_ind.function_mutation(self._variables)

    def __parent_selection__(self, island: str = "unique")-> tuple[Individual, Individual]:
        p1 = random.choice(self._populations[island])
        p2 = random.choice(self._populations[island])
        while p1 == p2:
            p1 = random.choice(self._populations[island])
            p2 = random.choice(self._populations[island])
        return p1, p2
    
    def __tournament_selection__(self, island: str = "unique")-> tuple[Individual, Individual]:
        tournament_size = self._population_size//5
        competitors = random.sample(self._populations, tournament_size)
        competitors.sort(key = lambda ind: ind.get_fitness())
        return competitors[0], competitors[1]

    def __selection__(self, island: str = "unique") -> Individual:
        return random.choice(self._populations[island])

    def __survival__(self, offsprings: list[Individual], island: str = "unique") -> list[Individual]:
        extended_population = self._populations[island] + offsprings
        extended_population.sort(key=lambda ind: ind.get_fitness())     # ORDERING FROM BEST TO WORSE
        #print([ind.SymRegTree._name for ind in extended_population])
        self._populations[island] = extended_population[:self._population_size]  # SURVIVAL SELECTION

    def __contamination__(self):
        # TODO: In order to perform contamination:
        #       select one random individual from each island
        #       Choose a island where we contaminate a number of individual


        ...

    def variable_checking(self, value):
        if value.shape[0] != len(self._variables):
            raise ValueError(f"This problem require {len(self._variables)} variables, but you passed only {value.shape[0]} variables !")
    
    def show_populations(self):
        for key, pop in self._populations.items():
            print(f"{str(key).upper()} island population:")
            for ind in pop:                
                ind.show_function()

    def show_individual(self, island: str = "unique"):
        self._populations[island][0].show_function()

    def deploy_population(self, val, island: str = "unique"):
        for ind in self._populations[island]:
            print("Computed value: ", ind.deploy_function(val))

    def start(self, x: np.ndarray[float], y: np.ndarray[float]):
        no_improvement_count = 0  ##  Generation without improvements

        ## Compute metrics
        [ind.compute_metrics(x, y) for _, pop in self._populations.items() for ind in pop] 

        best_ind_history = {key : list() for key in self._populations.keys()}

        #for e in tqdm(range(self._num_eras), desc="Era", leave=True, position = 0):
        for e in range(self._num_eras):
            print("era: ", e)
            #for i in tqdm(range(self._num_islands), desc="Island", leave=False, position = 1):
            for i in range(self._num_islands):
                print("isl: ", i)
                ## Work on the population of the selected island
                island = list(self._populations)[i]
                #for g in tqdm(range(self._num_generations), desc="Generation", leave=False, position = 2):
                for g in range(self._num_generations):
                    self._mutation_rate = max(0.05, 0.3 * (1 - g / self._num_generations))

                    # Troviamo il miglior individuo attuale
                    current_best = min(self._populations[island], key=lambda ind: ind.get_fitness())
    
                    # Check the improvement
                    if current_best.get_fitness() >= self.best_fitness:
                        self.no_improvement_count += 1
                    else:
                        self.no_improvement_count = 0
                        self.best_fitness = current_best.get_fitness()
    
                    if g < self._num_generations//3:      ##  Exploration
                        self.mutation_rate = max(0.3, min(0.4, self.mutation_rate * 1.1))
                    elif g < 2*self._num_generations//3:    ##  Balancing
                        self.mutation_rate = max(0.2, min(0.4, self.mutation_rate * (0.9 if self.no_improvement_count == 0 else 1.1)))
                    else:           ##  Exploitation
                        self.mutation_rate = max(0.05, min(0.5, self.mutation_rate * (0.8 if self.no_improvement_count == 0 else 1.3)))
                    
                    if self.no_improvement_count > 10:  ##  If stagnation more than 10 generation --> force max mutation
                        self.mutation_rate = 0.5 

                    print("gen: ", g)
                    offsprings = list()
                    #for o in tqdm(range(self._num_offsprings), desc="Offspring generated", leave=False, position = 3):
                    for o in range(self._num_offsprings):
                        #print("Off: ", o)
                        parent1, parent2 = self.__parent_selection__(island)    ## Usare la tecnica dell'UNPACKING
                        ind1, ind2 = self.__crossover__(parent1, parent2)

                        self.__mutation__(ind1, island)
                        self.__mutation__(ind2, island)
                        ind1.show_function()
                        ind2.show_function()
                        
                        print()

                        ind1.compute_metrics(x, y)
                        ind2.compute_metrics(x, y)
                        
                        if not ind1 == ind2:
                            offsprings.extend([ind1, ind2])
                        else:
                            offsprings.append(ind1)
 
                    self.__survival__(offsprings, island)
                    best_ind_history[island].append(deepcopy(self._populations[island][0]))
            ## TODO: Before the next era, I contaminate the island's individuals with a function from other island

        self.BEST_IND = self.__save_best_ind__().show_results()
        return best_ind_history

    def __save_best_ind__(self) -> Individual:    
        return min((ind for _, pop in self._populations.items() for ind in pop), key=lambda x: x.get_fitness())
    
    def get_best_ind(self) -> Individual:
        return self.BEST_IND
    
    def plot_fitness_history(self, history: list[Individual]):
        generations = list(range(1, (self._num_generations* self._num_eras)+1))
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
  # Longer function --> Penalty
  # Graph representation of function
  # Modificare l'interpretazione di livello del grafo -> 
  # Stalla perchè il crossover tende verso tutte funzioni uguali
  # Forzare ad avere tutte le variabili nell'albero della funzione
  # 3 tipi di mutazione: Mutazione var, val e funct; Mutazione Sottoalbero; Mutazione Strutturali
  # MUTATION RATE ADATTIVO !! --> self.mutation_rate = max(0.05, 0.3 * (1 - gen / max_generations))
  # ###