import numpy as np
import ipdb
class Interpreter:
    
    def __init__(self, data, model, idx_list, subset='test',use_bias=True) -> None:
        self.name = data.name
        self.train = data.train
        self.test = data.test
        self.n_timesteps = data.n_timesteps
        self.n_feats = data.n_feats
        self.model = model
        self.idx_list = idx_list
        self.subset = subset
        self.use_bias = use_bias
        self.label = data.train_target if subset == 'train' else data.test_target
        
        self.x_dict, self.x_label_dict, self.y_pred, self.weights_dict, self.intercept_dict, self.z_terms = self.interpret_instances()

        self.w_range = (np.min([w for _,w in self.weights_dict.items()]),np.max([w for _,w in self.weights_dict.items()]))
        self.z_range = (np.nanmin([z for _,z in self.z_terms.items()]),np.nanmax([w for _,w in self.z_terms.items()]))

    def getSparsity(self):
        # Sparsity metric: https://arxiv.org/pdf/2201.13291.pdf 
        z = [r for r in self.z_terms.values()]
        z = np.array(z)
        z = np.abs(z)
        # z_min = np.ones_like(z)*self.z_range[0]
        # z_max = np.ones_like(z)*self.z_range[1]
        # z_norm = (z - z_min)/(z_max-z_min)
        # z_mean = np.mean(z_norm,axis=(1,2)) 
        if(z[0].shape[-1]!=self.n_feats):
            z = np.repeat(z,self.n_feats,axis=-1)
        
        sparsity = np.sum(z>0.01,axis=(1,2))/(self.n_feats*self.n_timesteps)
        return np.mean(sparsity)
        


    def mlp_relu_interpretation(self, x):
        w_list, b_list = self.model.coefs_, self.model.intercepts_
        
        layers = range(len(w_list))
        layer_j = np.copy(x)
        activation_pattern = []
        for j in layers:
            if j == len(w_list) - 1:
                continue
            layer_j = layer_j @ w_list[j] + b_list[j]
            layer_j[layer_j <= 0] = 0
            activation_pattern.extend(np.where(layer_j > 0))
        
        for j in layers:
            if j == 0:
                feature_weights_layer = w_list[j][:,activation_pattern[j]]
                intercepts_layer = b_list[j][activation_pattern[j]]
            elif j > 0 and j < len(w_list) - 1:
                layer_j_w_active_input = w_list[j][activation_pattern[j-1],:]
                layer_j_w_active_output = layer_j_w_active_input[:,activation_pattern[j]]
                feature_weights_layer = feature_weights_layer @ layer_j_w_active_output
                intercepts_layer = intercepts_layer @ layer_j_w_active_output + b_list[j][activation_pattern[j]]
            elif j == len(w_list) - 1:
                layer_j_w_active_input = w_list[j][activation_pattern[j-1],:]
                feature_weights_layer = feature_weights_layer @ layer_j_w_active_input
                intercepts_layer = intercepts_layer @ layer_j_w_active_input + b_list[j]
        return feature_weights_layer, intercepts_layer

    def CAM_conv_interpretation(self,x):
        T = x.shape[1] 
        F,w_ = self.model.get_feature_map_weights_CAM(x)
        z = np.zeros((F.shape[0],1))
        for map in range(len(w_)):
            z = z + (F[...,map]*w_[map]).reshape((-1,1))
        
        return np.repeat(z,int(T/len(z)),axis=0)


    def conv_relu_mlp_interpretation(self,x,percent_act_info=0.8,onlyPosVals=False):
        x = x.reshape(1,self.n_timesteps,1,self.n_feats)
        maxPos,maxVal,featMap, weights,deConv = self.model.getConvInfo(x)
        # Max positions
        maxPos = np.argmax(featMap,axis=0) 
        # Extracting the dense weights 
        denseWeights,bias = self.mlp_relu_interpretation(maxVal)
        denseWeights =denseWeights.squeeze()
        #denseWeights = self.model.getDenseInfo(x) #TODO: change this for the unwrap algorithm
        # Filtering zero-activated neurons
        denseWeights = denseWeights*(maxVal>0)
        if onlyPosVals:
            denseWeights[denseWeights<0] = 0
        # Computing amount of filters needed to achive percent_act_info
        weights_actMaxVal = np.abs(maxVal*denseWeights)
        total_val_weights = np.sum(weights_actMaxVal)
        total_val_weights_percentage = total_val_weights*percent_act_info
        sorted_val_weights = np.sort(weights_actMaxVal)
        sorted_val_weights = sorted_val_weights[::-1]
        sorted_val_weights_cum_sum = np.cumsum(sorted_val_weights)
        # TODO: replace the loop for a vectorized expression
        for n in range(len(sorted_val_weights_cum_sum)):
            if sorted_val_weights_cum_sum[n] >= total_val_weights_percentage:
                break
        idx_top = np.argsort(weights_actMaxVal)[-n:]
        topPos = maxPos[idx_top]
        topVals = maxVal[idx_top]

        expl = np.zeros((self.n_timesteps,self.n_feats))
        for i in range(n):
            mask = np.zeros((1,deConv.input.shape[1],deConv.input.shape[2],deConv.input.shape[3]))
            mask[0,topPos[i],0,idx_top[i]] = 1
            expl = expl + denseWeights[idx_top[i]]*deConv.predict(mask).squeeze()
        
        return expl,bias




    def interpret_instances(self):
        all_x, all_labels,all_pred, all_weights, all_intercepts, z_terms = {}, {}, {}, {}, {},{}
        for idx in self.idx_list:
            label = self.label[idx]
            if self.subset == 'train':
                x = self.train[idx]
            else:
                x = self.test[idx]

            if self.model.model_type == 'relu-mlp':
                x_weights, x_intercept = self.mlp_relu_interpretation(x)
                y_pred = self.model.classifier.predict(x.reshape(1,-1))[0]
                z = x.reshape(x_weights.shape)*x_weights
                if self.use_bias:
                    z = z + np.ones_like(z)*(x_intercept/360)
                z = z.reshape((self.n_feats,self.n_timesteps)).T
                x_weights = x_weights.reshape((self.n_feats,self.n_timesteps)).T
                x = x.reshape((self.n_feats,self.n_timesteps)).T
            
            elif self.model.model_type == 'conv-relu-mlp':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                x_weights, x_intercept = self.conv_relu_mlp_interpretation(x,1)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.squeeze()
                z = x_weights*x
                if self.use_bias:
                    z = z + np.ones_like(z)*(x_intercept/(self.n_feats*self.n_timesteps)) #TODO: set 360 to a variable 

            elif self.model.model_type == 'CAM-conv':
                x = x.reshape((-1,self.n_feats,self.n_timesteps))
                x = np.transpose(x,(0,2,1)).reshape(-1,self.n_timesteps,1,self.n_feats)
                z = self.CAM_conv_interpretation(x)
                y_pred = self.model.classifier.predict(x)[0]
                x = x.squeeze()
                x_weights = 0
                x_intercept =0

            all_x[idx], all_labels[idx],all_pred[idx], all_weights[idx], all_intercepts[idx], z_terms[idx] = x, label,y_pred, x_weights, x_intercept, z      
        
        return all_x, all_labels, all_pred, all_weights, all_intercepts, z_terms