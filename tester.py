from address import load_obj, save_obj, results_obj
from data_constructor import Dataset
from model_constructor import Model
from interpreter_constructor import Interpreter

datasets = ['helix_1_random_time'] #'circles','moons','helix_1_start','helix_2_start','helix_1_random_time','helix_2_random_time','helix_random_feat_random_time','helix_matching_random_amp_feat_time'
train_fraction = 0.8
seed_int = 54321
idx_list = range(100)
subset = 'train'
model_type = 'relu-mlp' # 'relu-mlp', 'conv-relu-mlp'

if __name__ == '__main__':

    for data_str in datasets:
        data = Dataset(data_str, train_fraction, seed_int)
        model = Model(seed_int, data, model_type)
        model.train_model()
        interpreter = Interpreter(data, model, idx_list, subset=subset)
        save_obj(results_obj, interpreter, f'{data_str}_interpreter_{len(idx_list)}.pkl')