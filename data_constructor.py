import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons 
from address import dataset_dir

class Dataset:

    def __init__(self, dataset_name, train_fraction, seed, n=1000) -> None:
        self.name = dataset_name
        self.train_fraction = train_fraction
        self.seed = seed
        self.n = n
        self.n_timesteps         = 0
        self.n_feats             = 0
        self.train, self.train_target, self.test, self.test_target = self.load_dataset()

    
    def helix(self):
        dataset = pd.read_csv(f'{dataset_dir}/helix/{self.name}.csv', index_col=0)
        dataset, target = dataset.iloc[:,:-1].values, dataset['label'].values
        self.n_feats=5
        self.n_timesteps= 72
        return dataset, target
    def blink_EEG(self):
        dataset = pd.read_csv(f'{dataset_dir}/EEG-Blink-dataset/{self.name}.csv', index_col=0)
        dataset,target = dataset.iloc[:,:-1].values,dataset.iloc[:,-1].values 
        self.n_feats=3
        self.n_timesteps=510
        return dataset, target

    def load_dataset(self):
        np.random.seed(self.seed)
        if self.name == 'circles':
            dataset, target = make_circles(n_samples=self.n, factor=0.5, noise=0.1, random_state=self.seed)
        elif self.name == 'moons':
            dataset, target = make_moons(n_samples=self.n, noise=0.1, random_state=self.seed)
        elif 'helix' in self.name:
            dataset, target = self.helix()
        elif 'blink_EEG' in self.name:
            dataset, target = self.blink_EEG()
        train, test = dataset[:int(self.train_fraction*len(dataset)),:], dataset[int(self.train_fraction*len(dataset)):,:]
        train_target, test_target = target[:int(self.train_fraction*len(target))], target[int(self.train_fraction*len(target)):]
        return train, train_target, test, test_target
        