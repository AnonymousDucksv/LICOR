from address import load_obj, save_obj, results_obj
from data_constructor import Dataset
from model_constructor import Model
from interpreter_constructor import Interpreter
import ipdb

datasets = ['blink_EEG'] #'circles','moons','helix_1_start','helix_2_start','helix_1_random_time','helix_2_random_time','helix_random_feat_random_time','helix_matching_random_amp_feat_time', 'blink_EEG'
train_fraction = 0.8
seed_int = 54321
idx_list = range(100)
subset = 'train'
model_type = 'CAM-conv' # 'relu-mlp', 'conv-relu-mlp', 'CAM-conv'


if __name__ == '__main__':

    # Test params CNN
    # params = {'solver': 'adam',
    #     'hidden_layer_sizes': (200,100,50),
    #     "size_time_kernel":6,
    #     'n_kernels':32,
    #     'activation': 'relu'}

    for data_str in datasets:
        data = Dataset(data_str, train_fraction, seed_int)
        model = Model(seed_int, data, model_type)
        model.train_model()

        # # Probar la interpretabilidad de las convolucionales
        # #maxPos, maxVal,mapAct, weights, deConv = model.getConvInfo(model.X_test[2,...].reshape(1,72,1,5))

        # F,w = model.get_feature_map_weights_CAM(model.X_test[2,...].reshape(1,72,1,5))

        
        interpreter = Interpreter(data, model, idx_list, subset='test')

        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.ion()
        # idxs = np.random.permutation(np.arange(model.X_test.shape[0]))[:10]
        # import matplotlib.pyplot as plt
        # plt.ion()
        # sample = model.X_train[idxs,...]
        # sampl_y = model.y_train[idxs,...]
    
        # plt.figure()
        # for ii,idx in enumerate(idxs):
        #     plt.subplot(len(idxs),1,ii+1)
        #     for f in range(data.n_feats):
        #         plt.plot(sample[ii,:,:,f])
        #     plt.title(f"Label {sampl_y[ii]}")
