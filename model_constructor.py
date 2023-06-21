
import numpy as np
import pandas as pd
import ast
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,roc_auc_score
from address import results_grid_search
from tensorflow.keras import layers, losses, metrics, utils, callbacks,backend
from tensorflow.keras import Model as kerasModel
import tensorflow as tf
import ipdb


def f1_loss(y_true, y_pred):
    
    tp = backend.sum(backend.cast(y_true*y_pred, 'float'), axis=0)
    tn = backend.sum(backend.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = backend.sum(backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2*p*r / (p+r+backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - backend.mean(f1)

class Model:

    def __init__(self, seed, data, model_type, params=None) -> None:
        self.seed = seed
        self.data = data
        self.name = data.name
        self.n_feats = self.data.n_feats
        self.n_timesteps = self.data.n_timesteps
        self.model_type = model_type
        if params is None:
            self.params = self.read_params()
        else:
            self.params = params
    
    def create_conv_CAM(self,n_timesteps:int, n_feats:int, size_time_kernel, activations='relu'):
        """Method to create the CNN + CAM keras model"""
        input_layer   = layers.Input(dtype = tf.float32,shape=(n_timesteps,1,n_feats),name='input')
        # features_layer = layers.DepthwiseConv2D(kernel_size = (size_time_kernel,1),
        #                                         strides     = (size_time_kernel,size_time_kernel),
        #                                         padding     = 'valid',
        #                                         activation  = activations,
        #                                         use_bias    = False,
        #                                         name = "Conv1")(input_layer)
        features_layer = layers.Conv2D(filters     = 16,
                                       kernel_size = (size_time_kernel,1),
                                       strides     = (1,1),
                                       padding     = 'same',
                                       activation  = activations,
                                       name = "Conv1")(input_layer)
        features_layer = layers.MaxPool2D(pool_size=(2,1))(features_layer)
        features_layer = layers.Conv2D(filters     = 32,
                                       kernel_size = (size_time_kernel,1),
                                       strides     = (1,1),
                                       padding     = 'same',
                                       activation  = activations,
                                       name = "Conv2")(features_layer)
        features_layer = layers.MaxPool2D(pool_size=(2,1))(features_layer)
        features_layer = layers.Conv2D(filters     = 64,
                                       kernel_size = (size_time_kernel,1),
                                       strides     = (1,1),
                                       padding     = 'same',
                                       activation  = activations,
                                       name = "ConvOutput")(features_layer)
        feature_map_length = features_layer.shape[1]
        GAP = layers.GlobalAveragePooling2D()(features_layer)
        y = layers.Dense(units = 1,activation='sigmoid' ,name = 'output')(GAP)
        classifier = kerasModel(inputs = input_layer, outputs=y)
        classifier.summary()
        classifier.compile(optimizer='adam', loss=losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC()])
        # Models for interpretability
        self.featureMapModel = kerasModel(inputs = input_layer, outputs=features_layer)
        return classifier
        
        
    
    def create_conv_network(self, n_timesteps:int, n_feats:int, size_time_kernel,n_kernels,hidden_layers_sizes:tuple, activations='relu'):
        """Method to create the CNN keras model"""
        input_layer   = layers.Input(dtype = tf.float32,shape=(n_timesteps,1,n_feats),name='input')
        # features_layer = layers.DepthwiseConv2D(kernel_size = (size_time_kernel,1),
        #                                         strides     = (size_time_kernel,size_time_kernel),
        #                                         padding     = 'valid',
        #                                         activation  = activations,
        #                                         use_bias    = False,
        #                                         name = "Conv1")(input_layer)


        features_layer = layers.Conv2D(filters     = n_kernels,
                                       kernel_size = (size_time_kernel,1),
                                       strides     = (1,1),
                                       padding     = 'valid',
                                       activation  = activations,
                                       use_bias    = False,
                                       #kernel_constraint=  tf.keras.constraints.non_neg(),
                                       name = "Conv1")(input_layer)
        feature_map_length = features_layer.shape[1]
        maxPool,arg = tf.nn.max_pool_with_argmax(features_layer,(1,feature_map_length,1,1),(1,feature_map_length,1,1),padding='VALID')
        x = layers.Flatten()(maxPool)
        
        for ii,hidden_layer_size in enumerate(hidden_layers_sizes):
            x = layers.Dense(units = hidden_layer_size,activation='relu',name=f"Dense{ii}")(x)
        y = layers.Dense(units = 1,activation='sigmoid' ,name = 'output')(x)
        classifier = kerasModel(inputs = input_layer, outputs=y)
        classifier.summary()
        classifier.compile(optimizer='adam', loss=losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.AUC()])
        
        # Models for interpretability
        
        self.deconvInput = layers.Input(dtype = tf.float32,shape=features_layer.shape[1:],name='DeConvInput')
        self.deConvLayer = layers.Conv2DTranspose(filters     = n_feats,
                                       kernel_size = (size_time_kernel,1),
                                       strides     = (1,1),
                                       padding     = 'valid',
                                       use_bias    = False,
                                       #kernel_constraint=  tf.keras.constraints.non_neg(),
                                       name = "DeConv1")
        
        self.deConv      = kerasModel(inputs = self.deconvInput,outputs=self.deConvLayer(self.deconvInput))
        self.argMaxModel = kerasModel(inputs=input_layer,outputs = arg)
        self.maxVals    = kerasModel(inputs=input_layer,outputs = maxPool)
        self.featMap    = kerasModel(inputs=input_layer,outputs = features_layer)
        return classifier

    def read_params(self):
        if self.model_type == 'relu-mlp':
            grid_search_df = pd.read_csv(results_grid_search+'grid_search.csv',index_col=0)
        elif self.model_type == 'conv-relu-mlp':
            grid_search_df = pd.read_csv(results_grid_search+'grid_searchCNN.csv',index_col=0)
        elif self.model_type == 'CAM-conv':
            grid_search_df = pd.read_csv(results_grid_search+'grid_searchCNN_CAM.csv',index_col=0)
        params = ast.literal_eval(grid_search_df.loc[self.name, 'params'])[0]
        return params
    
    def train_model(self):
        """Method to create and train the model"""
        if self.model_type == 'relu-mlp':
            self.classifier = MLPClassifier(hidden_layer_sizes=self.params['hidden_layer_sizes'], solver=self.params['solver'], random_state=self.seed, max_iter=1000)
            X_train, y_train = self.data.train, self.data.train_target
            X_test, y_test = self.data.test, self.data.test_target
            self.classifier.fit(X_train, y_train)
            self.test_auc = roc_auc_score(y_test, self.classifier.predict(X_test))
            print(f'Test AUC score {self.data.name}: {self.test_auc}')
            self.coefs_,self.intercepts_ = self.classifier.coefs_,self.classifier.intercepts_
            
        elif self.model_type == 'conv-relu-mlp':
            """
            The convolutional ReLU MLP has two sequential stages: (1) For time and (2) For channels (order can be changed).
            Here we first define the kernel characteristics for the first stage and second stages. These are also ReLU activated.
            Since the dataset is originally flattened, the dataset for the training and testing here has to be reshaped.
            """
            n_timesteps         = self.data.n_timesteps
            n_feats             = self.data.n_feats

            X_train             = self.data.train.reshape((-1,n_feats,n_timesteps))
            self.X_train             = np.transpose(X_train,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
            self.y_train             = self.data.train_target

            X_test              = self.data.test.reshape((-1,n_feats,n_timesteps))
            self.X_test              = np.transpose(X_test,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
            self.y_test              = self.data.test_target
            
            size_time_kernel    = self.params.get('size_time_kernel',6)
            n_kernels           = self.params.get('n_kernels',7)
            hidden_layers_sizes = self.params.get('hidden_layers_sizes',(100,30,10))

            self.classifier = self.create_conv_network(n_timesteps,
                                                 n_feats,
                                                 size_time_kernel,                                                 
                                                 n_kernels,
                                                 hidden_layers_sizes)
            ES_cb = callbacks.EarlyStopping( monitor="val_loss",
                                            min_delta=0.01,
                                            patience=5,                                            
                                            baseline=None,
                                            restore_best_weights=True)
            self.classifier.fit(self.X_train, self.y_train, batch_size=32,epochs=100, validation_data=(self.X_test, self.y_test),callbacks=[ES_cb])
            test_loss, self.test_auc = self.classifier.evaluate(self.X_test, self.y_test, verbose=2)
            print(f'Test AUC score {self.data.name}: {self.test_auc}')

            # Getting weights and bias for the unwrap
            dense_layers = [l for l in self.classifier.layers if ("Dense" in l.name) or ("output" in l.name)]
            self.coefs_ = [l.get_weights()[0] for l in dense_layers]
            self.intercepts_ = [l.get_weights()[1] for l in dense_layers]

        elif self.model_type == "CAM-conv":
            n_timesteps         = self.data.n_timesteps
            n_feats             = self.data.n_feats

            X_train             = self.data.train.reshape((-1,n_feats,n_timesteps))
            X_train             = np.transpose(X_train,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
            y_train             = self.data.train_target

            X_test              = self.data.test.reshape((-1,n_feats,n_timesteps))
            self.X_test         = np.transpose(X_test,(0,2,1)).reshape(-1,n_timesteps,1,n_feats)
            self.y_test         = self.data.test_target

            size_time_kernel    = self.params.get('size_time_kernel',3)

            self.classifier = self.create_conv_CAM(n_timesteps, n_feats, size_time_kernel, activations='relu')

            ES_cb = callbacks.EarlyStopping( monitor="val_loss",
                                            min_delta=0.01,
                                            patience=5,                                            
                                            baseline=None,
                                            restore_best_weights=True)
            self.classifier.fit(X_train, y_train, batch_size=32,epochs=100, validation_data=(self.X_test, self.y_test),callbacks=[ES_cb])
            test_loss, self.test_auc = self.classifier.evaluate(self.X_test, self.y_test, verbose=2)
            print(f'Test AUC score {self.data.name}: {self.test_auc}')

        return self.classifier,self.test_auc

    def get_feature_map_weights_CAM(self,x):
        """ Method for getting the kernel and the time position of the max activation
            PARAMS
            x --> Input numpy array
            OUTPUT
            [F,w]
            The the entire feature map and the weights of the last layer.
        """      
        F = self.featureMapModel.predict(x)
        w = self.classifier.get_layer('output').get_weights()[0]
        return F.squeeze(),w.squeeze()
        
        




    def getConvInfo(self,x):
        """ Method for getting the kernel and the time position of the max activation
            PARAMS
            x --> Input numpy array
            OUTPUT
            [maxPos,maxVal,weights]
            The positions of the kernel, and the value of the max activation in the feature map and the kernels' weights
        """    
        self.deConvLayer.set_weights(self.classifier.get_layer('Conv1').get_weights())
        maxPos = self.argMaxModel.predict(x)
        maxVal = self.maxVals.predict(x)
        mapAct = self.featMap.predict(x)
        weights = self.classifier.get_layer(name="Conv1").weights


        return maxPos.squeeze(), maxVal.squeeze(),mapAct.squeeze(), weights[0].numpy().squeeze(), self.deConv
    
    def getDenseInfo(self,x):
        return self.classifier.get_layer(name="output").weights[0].numpy().squeeze()