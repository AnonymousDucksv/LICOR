import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import itertools
from address import results_grid_search
from data_constructor import Dataset
from tester import datasets, train_fraction, seed_int

path_here = os.path.abspath('')
np.random.seed(seed_int)
results = pd.DataFrame(index=datasets, columns=['params','F1'])
number_neurons = [10, 20, 50, 100, 200, 300]
number_layers = 6
all_possible_architectures = []
for i in range(3, number_layers):
    possible_architectures_i = list(itertools.product(number_neurons, repeat=i))
    all_possible_architectures.extend(possible_architectures_i)

param_grid = {'hidden_layer_sizes':all_possible_architectures, 'activation':['relu'], 'solver':['lbfgs','sgd','adam']}

for data_str in datasets:
    data = Dataset(data_str, train_fraction, seed_int)
    clf_search = RandomizedSearchCV(MLPClassifier(), param_distributions=param_grid, scoring='f1', cv=5, verbose=2, n_iter=int(len(all_possible_architectures*3)*0.05))
    clf_search.fit(data.train, data.train_target)
    results.loc[data_str,'params'] = [clf_search.best_params_]
    results.loc[data_str,'F1'] = clf_search.best_score_
results.to_csv(f'{results_grid_search}grid_search.csv',mode='a')
