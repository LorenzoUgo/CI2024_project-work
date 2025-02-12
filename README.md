# CI2024_Project_Work
The project work aimed to solve 8 Symbolic Regression problems.

***Symbolic Regression*** is a machine learning technique that discovers mathematical expressions to describe relationships in data, without assuming a predefined model structure. Unlike traditional regression, it explores possible equations using Genetic Programming, leading to interpretable models that reveal underlying patterns.

The dataset are:
- 2 variables, 1000 points
- 1 variable, 500 points
- 3 variables, 5000 points
- 3 variables, 5000 points
- 2 variables, 5000 points
- 2 variables, 5000 points
- 2 variables, 5000 points
- 6 variables, 5000 points
### Comment section

This project implements a Genetic Algorithm (GA) for Symbolic Regression, leveraging tree-based representations to evolve mathematical expressions.

Feature:
- **Tree-based Representation**: Individuals are structured as expression trees.

- **Crossover & Mutation**: Genetic operators allow recombination and diversification.

- **Multi-Island Approach**: Support for multiple islands with different function sets.

- **Fitness Evaluation**: Uses Mean Squared Error (MSE) with penalties for complexity.

- **Selection Strategies**: Tournament selection and random selection available.

- **Dynamic Mutation Rate**: Adjusts over time to balance exploration and exploitation.

- **Migration & Contamination**: Ensures diversity and prevents stagnation.

***Key Classes***
***Genetic_Algorithm*** -> Handles the entire evolutionary process:

- *start()*: Runs the genetic algorithm.

- *get_best_ind()*: Returns the best individual found.

- *show_populations()*: Displays all individuals in the population.

- *plot_fitness_history()*: Plots fitness evolution.

***Individual*** -> Represents a single candidate solution:

- *deploy_function()*: Evaluates the expression tree.

- *compute_metrics()*: Computes fitness based on MSE.

- *gene_crossover()*: Performs subtree crossover.

- *function_mutation()*: Applies function mutation.

***Node*** -> Represents a node in the symbolic expression tree:

- *apply()*: Evaluates the function.

- *insert_intermediate_node()*: Adds an intermediate function.

- *remove_and_merge()*: Removes a node and merges children.

***Genetic Operators***

- *Crossover*: Swaps subtrees between two individuals.

- *Mutation*: Modifies nodes by changing values, functions, or structure.

- *Survival Selection*: Retains best individuals in the population.

- *Migration*: Moves individuals between islands for diversity.

- *Contamination*: Injects new individuals to prevent stagnation.
