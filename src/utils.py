##
# All the supporting function needed for my algorithm.
# ##
import numpy as np
import warnings

import src.s315734 as s315734
import warnings

from src.individual import Individual

warnings.simplefilter("ignore", category=RuntimeWarning)


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

    x_shuffled = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
    y_shuffled = np.apply_along_axis(np.random.permutation, axis=1, arr=y)


    _train = (x_shuffled[:, :idx_split], y_shuffled[:idx_split])
    _test = (x_shuffled[:, idx_split:], y_shuffled[idx_split:])

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
  # Graph representation of function
  # Modificare l'interpretazione di livello del grafo -> 
  # ###