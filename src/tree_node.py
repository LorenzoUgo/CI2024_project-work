##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
import itertools
from typing import Callable
import numbers
import warnings
import random
import types

import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.exp2, np.log, np.log10, np.log2],       ##  , np.expm1, np.log1p
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin],    ##    , np.hypot
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.absolute, np.reciprocal],     ##  , np.square
}

numpy_cost = {np.e, np.pi, np.euler_gamma, np.finfo(float).eps}

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
        if not nodes:
            type = "all"
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

    def get_size(self) -> int:
        return 1 + sum(child.get_size() for child in self._successor)
    
    '''def draw(self):
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None'''
