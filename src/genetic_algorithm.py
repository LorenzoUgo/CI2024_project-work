##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
import itertools
import warnings
import random
import matplotlib.pyplot as plt
from copy import deepcopy

import warnings

from src.individual import Individual
from src.tree_node import Node

warnings.simplefilter("ignore", category=RuntimeWarning)

numpy_funct = {
    "arithmetic":       [np.add, np.multiply, np.divide, np.subtract, np.floor_divide, np.power, np.mod],
    "expandlog":        [np.exp, np.exp2, np.log, np.log10, np.log2],       ##  , np.expm1, np.log1p
    "trigonometric":    [np.sin, np.cos, np.tan, np.arccos, np.arctan, np.arcsin],    ##    , np.hypot
    "hyperbolic":       [np.sinh, np.cosh, np.tanh, np.arccosh, np.arctanh, np.arcsinh],
    "miscellaneus":     [np.sqrt, np.cbrt, np.absolute, np.reciprocal],     ##  , np.square
}

numpy_cost = {np.e, np.pi, np.euler_gamma, np.finfo(float).eps}


class Genetic_Algorithm:
    _population_size: int
    _num_offsprings: int
    _num_islands: int
    _populations: dict[str: list[Individual]]
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

    def __inject_new_individuals__(self, island: str = "unique", rate: float = 0.1):
        """Add random individuals to avoid stagnation"""
        num_new = int(self._population_size * rate)
        return [Individual(self.__random_tree__(island=island, max_depth=len(self._variables))) for _ in range(num_new)]

    def __mutation__(self, ind: Individual, island: str = "unique") -> Individual: 
        if random.random() > self._mutation_rate :
            return # No mutation
        
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
            ind.function_mutation(self._variables, mutation_rate=self._mutation_rate, island=island)
            #   return new_ind.function_mutation(self._variables)

    def __parent_selection__(self, island: str = "unique")-> tuple[Individual, Individual]:
        p1 = random.choice(self._populations[island])
        p2 = random.choice(self._populations[island])
        attempts = 0
        while p1 == p2 and attempts < 5:
            p2 = random.choice(self._populations[island])
            attempts += 1
    
        if p1 == p2:
            self.__mutation__(p2, island)
        return p1, p2
    
    def __tournament_selection__(self, island: str = "unique")-> tuple[Individual, Individual]:
        tournament_size = self._population_size//5
        competitors = random.sample(self._populations[island], tournament_size)
        competitors.sort(key = lambda ind: ind.get_fitness())
        return competitors[0], competitors[random.randint(1, len(competitors) - 1)]

    def __selection__(self, island: str = "unique") -> Individual:
        return random.choice(self._populations[island])

    def __survival__(self, offsprings: list[Individual], island: str = "unique") -> list[Individual]:
        extended_population = self._populations[island] + offsprings
        extended_population.sort(key=lambda ind: ind.get_fitness())     # ORDERING FROM BEST TO WORSE
        #print([ind.SymRegTree._name for ind in extended_population])
        self._populations[island] = extended_population[:self._population_size]  # SURVIVAL SELECTION

    def __contamination_1_island__(self, x: np.ndarray[float], y: np.ndarray[float], island: str = "unique" ):
        new_inds = self.__inject_new_individuals__(island=island)
        [ind.compute_metrics(x, y) for ind in new_inds] 

        len_new = len(new_inds)

        keep_ind = self._population_size-len_new
        self._populations[island] = self._populations[island][:keep_ind] + new_inds

    def __migration__(self, contamination_rate: float = 0.25):
        # TODO: In order to perform contamination:
        #       select one random individual from each island
        #       Choose a island where we contaminate a number of individual
        num_migrants = max(1, int(self._population_size * contamination_rate))

        islands = list(self._populations.keys())
        random.shuffle(islands)

        for i in range(len(islands)):
            source_island = islands[i%len(islands)]
            target_island = islands[(i + 1)%len(islands)]
            
            random.shuffle(self._populations[source_island])
            migrants = [self._populations[source_island].pop() for _ in range(num_migrants)]

            self._populations[target_island].extend(migrants)

    def __compute_population_diversity__(self, island: str = "unique"):
        pop = self._populations[island]

        if len(pop) <= 1:
            return 0, 0

        unique_trees_rate = len(set(ind for ind in pop)) / self._population_size

        # Calcola la varianza della fitness
        fitness_values = [ind.get_fitness() for ind in pop]
        fitness_variance = np.var(fitness_values)

        return unique_trees_rate, fitness_variance

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
        self.best_fitness = float("inf")

        ## Compute metrics
        [ind.compute_metrics(x, y) for _, pop in self._populations.items() for ind in pop] 

        best_ind_history = {key : list() for key in self._populations.keys()}

        #for e in tqdm(range(self._num_eras), desc="Era", leave=True, position = 0):
        for e in range(self._num_eras):
            print("era: ", e)
            print([len(x[1]) for  x in self._populations.items()])
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
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                        self.best_fitness = current_best.get_fitness()
    
                    if g < self._num_generations//3:      ##  Exploration
                        self._mutation_rate = max(0.3, min(0.4, self._mutation_rate * 1.1))
                    elif g < 2*self._num_generations//3:    ##  Balancing
                        self._mutation_rate = max(0.2, min(0.4, self._mutation_rate * (0.9 if no_improvement_count == 0 else 1.1)))
                    else:       ##  Exploitation
                        self._mutation_rate = max(0.05, min(0.5, self._mutation_rate * (0.8 if no_improvement_count == 0 else 1.3)))
                    
                    if no_improvement_count > 10:  ##  If stagnation more than 10 generation --> force max mutation
                        self._mutation_rate = 0.5 

                    print("gen: ", g)
                    offsprings = list()
                    #for o in tqdm(range(self._num_offsprings), desc="Offspring generated", leave=False, position = 3):
                    for o in range(self._num_offsprings):
                        #print("Off: ", o)
                        parent1, parent2 = self.__tournament_selection__(island)    ## Usare la tecnica dell'UNPACKING
                        ind1, ind2 = self.__crossover__(parent1, parent2)

                        self.__mutation__(ind1, island)
                        self.__mutation__(ind2, island)
                        ##  ind1.show_function()
                        ##  ind2.show_function()
                        ##  
                        ##  print()

                        ind1.compute_metrics(x, y)
                        ind2.compute_metrics(x, y)
                        
                        if not ind1 == ind2:
                            offsprings.extend([ind1, ind2])
                        else:
                            offsprings.append(ind1)
 
                    self.__survival__(offsprings, island)
                    best_ind_history[island].append(deepcopy(self._populations[island][0]))

                ##TODO : Elaborate any low of diversity in poulation

            if self._num_islands > 1:
                self.__migration__() 
            else:
                self.__contamination_1_island__(x, y)
            ## TODO: Before the next era, I contaminate the island's individuals with a function from other island

        self.BEST_IND = self.__save_best_ind__().show_results()     ## TODO: Fix the really best
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
