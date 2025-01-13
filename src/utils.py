##
# All the supporting function needed for my algorithm.
# ##
import numpy as np

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