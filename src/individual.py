import numpy as np
import itertools
import numbers
import random
from functools import total_ordering
from copy import deepcopy

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

from src.tree_node import Node


numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.exp2, np.log, np.log10, np.log2],       ##  , np.expm1, np.log1p
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin],    ##    , np.hypot
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.absolute, np.reciprocal],     ##  , np.square
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
    
    def __compute_fitness__(self, penalty: float = 0.2, num_vars: int = 1):
        ''' MSE + f_length + missing_vars '''
        tree_size = self.SymRegTree.get_size()
        missing_vars = self.get_missing_variables(num_vars)  # Conta quante variabili mancano

        return self.MSE + penalty * tree_size + 100*(missing_vars/num_vars)     ##(0.5*(missing_vars/num_vars) + 1)*
    
    def __select_random_subtree__(self, avoid_root: bool = False, size_bias: bool = False) -> "Node":
        nodes = self.SymRegTree.get_all_nodes()
        ### consenti o root o foglia --> 
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
        self.fitness = self.__compute_fitness__(num_vars=x.shape[0])
    
    def get_missing_variables(self, num_vars: int = 1):
        tree_vars = {node._value for node in self.SymRegTree.get_all_nodes(type="leaf") if isinstance(node._value, str)}
        return num_vars - len(set(tree_vars))
    
    def show_function(self):
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
