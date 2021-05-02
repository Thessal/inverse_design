from tmm import *
import itertools
import math
import numpy as np
import pandas as pd
import os
# from tqdm import tqdm
from p_tqdm import p_map
from scipy.spatial.distance import squareform, pdist
import keras
import hashlib
import random


def convert(number: float, type_str: str, convert_str: str, length_unit_in_meter=150e-9, n=1):
    #
    # Utility function for unit conversion
    #
    from math import pi
    k = None  # frequency
    l = None  # wavelength
    c = 3e8 / n
    a = length_unit_in_meter
    units_f = {"THz": 1e-12 * c / a, "Hz": c / a, "c/a": 1, "/m": 2 * pi / a, "/cm": 2 * pi / a / 100,
               "/nm": 2 * pi / a / 1e9, "/a": 2 * pi}
    units_l = {"m": a, "nm": 1e9 * a, "a": 1}
    units = {**units_f, **units_l}
    if (type_str not in units) or (convert_str not in units):
        raise NotImplementedError("Unit is not recognized")
    elif (type_str in units_f) == (convert_str in units_f):
        return units[convert_str] / units[type_str] * number
    else:
        return units[convert_str] * units[type_str] / number


def generate_structure(config: dict, params: list):
    return {"d": [np.inf, *params, np.inf],
            "n": [1.0] + ([config["n1"], config["n2"]] * ((config["layer_count"] + 1) // 2))[:config["layer_count"]] + [
                1.0]}


def generate_structure_all(config):
    #
    # Define uniform search parameter space, No material dispersion
    #
    search_size = config["simulation_count"]
    resolution = math.ceil(search_size ** (1 / config["layer_count"]))
    space_size = resolution ** config["layer_count"]
    ignore_below = 1e-12
    _digits = -math.floor(math.log(ignore_below, 10))
    d = [round(x, _digits) for x in np.linspace(config["d_min"], config["d_max"], resolution)]
    params = itertools.product(*([d] * config["layer_count"]))
    if search_size < space_size * 0.8 :
        print(f"Warning : search space skewed ({search_size}/{space_size})")
    outputs = [generate_structure(config, p) for p in params]
    random.seed(42)
    random.shuffle(outputs)
    for structure in outputs[:search_size]:
        yield structure
#     for i, p in enumerate(params):
#         if i == config["simulation_count"]: break
#         yield generate_structure(config, p)


def get_parameter_count(config: dict):
    return config["layer_count"]


def get_parameter_names(config: dict):
    return ['param_' + str(i) for i in range(get_parameter_count(config))]


def get_parameter(structure: dict):
    return structure["d"][1:-1]


def calculate_spectrum(structure: dict, ks: list = np.linspace(0.0104720, 0.00628319, num=200)):
    n_list = structure["n"]
    d_list = [x * 1e9 for x in structure["d"]]
    spectrum = [coh_tmm('s', n_list, d_list, 0, 1 / k)['R'] for k in ks]
    return spectrum


def calculate_spectrum_all(config):
    #
    # Precalculate spectrum (forward looking!)
    # row = structure
    # column = wavelength
    #
    filename = f"pkl/sim_{hashlib.md5(str(config).encode()).hexdigest()}.pkl"
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
    else:
        #         structures = generate_structure(config["simulation_count"],layers=config["layer_count"])
        structures = generate_structure_all(config)
        f1 = convert(config["spectral_range"][1], "c/a", "/nm")
        f2 = convert(config["spectral_range"][0], "c/a", "/nm")
        ks = np.linspace(f1, f2, config["spectral_resolution"])
        # results = [ get_parameter(s) + calculate_spectrum(s, ks) for s in tqdm(structures, total = config["simulation_count"])]
        results = p_map(lambda x: get_parameter(x) + calculate_spectrum(x, ks), list(structures))
        columns = get_parameter_names(config) + list(ks)
        df = pd.DataFrame(results, columns=columns)
        df.to_pickle(filename)
    return df


def filter_similar_spectrum(config, df, thres_corr=0.2, thres_ratio=None, plot=False):
    #
    # Filter spectrum by euclidian distance
    # However it is still overdetermined 
    # 
    df.reset_index(drop=True, inplace=True)
    assert (0 <= thres_corr <= 1)
#     assert (df.shape == (config["simulation_count"], get_parameter_count(config) + config["spectral_resolution"]))
    assert (df.shape[1] == get_parameter_count(config) + config["spectral_resolution"])
    distance = pdist(df.iloc[:, get_parameter_count(config):])
    dist_mat = pd.DataFrame(np.triu(squareform(distance)))
    maxcorr = dist_mat.max()
    thres = thres_corr 
    if thres_ratio : 
        assert (0 <= thres_ratio <= 1)
        thres = np.quantile(maxcorr, thres_ratio)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.matshow(dist_mat, cmap='Greys', vmin=0, vmax=1)
        plt.figure()
        plt.hist(distance,bins=50)
        plt.figure()
        maxcorr.hist(bins=50)
        plt.figure()
        plt.matshow(dist_mat.loc[maxcorr < thres, maxcorr < thres], cmap='Greys', vmin=0, vmax=1)
    return df[maxcorr < thres]


def train_test_split(df, ratio):
    train_count = math.floor(df.shape[0] * ratio)
    #df.reindex(np.random.permutation(df.index),inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    df_train = df.iloc[:train_count, :]
    df_train.reset_index(inplace=True, drop=True)
    df_test = df.iloc[train_count:, :]
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf


class inverse_model:
#     normalizer = Normalization()
    history_forward = None
    history_backward = None

    def __init__(self, config):
        # Constants
        self.config = config
        f1 = convert(config["spectral_range"][1], "c/a", "/nm")
        f2 = convert(config["spectral_range"][0], "c/a", "/nm")
        self.ks = np.linspace(f1, f2, config["spectral_resolution"])
        
        # Input data spec for normalizatino
        self.mean = 0.5 * (self.config["d_max"] + self.config["d_min"])
        self.std = 0.5 * (self.config["d_max"] - self.config["d_min"])
        spectrum_max = 1.0 
        spectrum_min = 0.0
        self.mean_y = 0.5 * (spectrum_max + spectrum_min)
        self.std_y = 0.5 * (spectrum_max - spectrum_min)

        self.model_forward = Sequential([
            keras.Input(shape=(get_parameter_count(config))),
            Dense(config["spectral_resolution"] * 2.5, activation='tanh'),
            Dense(config["spectral_resolution"] * 2.5, activation='tanh'),
            Dense(config["spectral_resolution"] * 2.5, activation='tanh'),
            Dense(config["spectral_resolution"], activation='tanh'),
        ])
        
        '''
        def my_loss_fn(y_true, y_pred):
            squared_difference = tf.square(y_true - y_pred)
            tf.print("---")
            tf.print(y_true, y_pred)
            tf.print(y_true.shape, y_pred.shape, tf.reduce_mean(squared_difference, axis=-1).shape)
            tf.print(tf.reduce_mean(squared_difference, axis=-1) )
            tf.print("---")
            return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
        '''
        self.model_forward.compile(optimizer='adam',
                                   loss=tf.keras.losses.mse,
                                   # loss=my_loss_fn,
                                   metrics=['mse'])                                   
                                   # metrics=[my_loss_fn])
                                   #metrics=['accuracy'])

        self.model_backward = Sequential([keras.Input(shape=(config["spectral_resolution"])),
                                          Dense(config["spectral_resolution"] * 2.5, activation='tanh'),
                                          Dense(config["spectral_resolution"], activation='tanh'),
                                          Dense(get_parameter_count(config), activation='tanh'),
                                          ] + self.model_forward.layers)
        self.model_backward.compile(optimizer='adam',
                                    loss=tf.keras.losses.mse,
                                    metrics=['mse'])
                                    #metrics=['accuracy'])

        self.model_inverse = Sequential([keras.Input(shape=(config["spectral_resolution"]))]
                                        + self.model_backward.layers[:-len(self.model_forward.layers)])

    def summary(self):
        self.model_forward.summary()
        self.model_backward.summary()
        self.model_inverse.summary()

    def freeze(self):
        for l in self.model_forward.layers:
            l.trainable = False

    def train(self, df, train_epochs=10):
        # Prepare data 
        df_train, df_test = train_test_split(df,0.9)
        X = df_train.iloc[:, :get_parameter_count(self.config)] # structure
        Y = df_train.iloc[:, get_parameter_count(self.config):] # spectrum
        val_X = df_test.iloc[:, :get_parameter_count(self.config)]
        val_Y = df_test.iloc[:, get_parameter_count(self.config):]

        X = self._normalize(X,mean=self.mean,std=self.std)
        Y = self._normalize(Y,mean=self.mean_y,std=self.std_y)
        val_X = self._normalize(val_X,mean=self.mean,std=self.std)
        val_Y = self._normalize(val_Y,mean=self.mean_y,std=self.std_y)
        
        # forward train
        self.history_forward = self.model_forward.fit(x=X.values, y=Y.values, epochs=train_epochs, verbose = 0, validation_data =(val_X.values,val_Y.values) )
        #return

        # backward train
        self.freeze()
        self.history_backward = self.model_backward.fit(x=Y.values, y=Y.values, epochs=train_epochs, verbose = 0, validation_data = (val_Y.values,val_Y.values) )

    def show_history(self):
        import matplotlib.pyplot as plt
        if self.history_forward:
            print("Forward training train/test loss")
            plt.plot(self.history_forward.history['loss'])
            plt.plot(self.history_forward.history['val_loss'])
            plt.show()
        if self.history_backward:
            print("Tandem training train/test loss")
            plt.plot(self.history_backward.history['loss'])
            plt.plot(self.history_backward.history['val_loss'])
            plt.show()

    # Inverse design
    
    def _normalize(self,X,mean,std):
        # X : numpy array
        # std : (max-min)/2
        return (X - mean) / std

    def _denormalize(self, tensor,mean,std):
        assert(tensor.shape[0]==1)
        return list(np.array(tensor[0]) * std + mean)

    def design(self, spectrum):
        spectrum = [self._normalize(x,mean=self.mean_y,std=self.std_y) for x in spectrum]
        input_ = tf.reshape(tf.convert_to_tensor(spectrum), (1, self.config["spectral_resolution"]))
        output_ = self._denormalize(self.model_inverse(input_), mean=self.mean,std=self.std)
        #         output_structure = {"d" : [np.inf, *output_, np.inf],
        #                      "n": [1.0]+[1.4,2.1]*(len(output_)//2)+[1.0]}
        output_structure = generate_structure(self.config, output_)
        output_spectrum = calculate_spectrum(output_structure, ks=self.ks)
        return output_structure, output_spectrum

    def test(self, df, plot=False):
        test_idx = random.randint(0, df.shape[0] - 1)
        input_structure = df.iloc[test_idx, :][:get_parameter_count(self.config)].values
        input_spectrum = df.iloc[test_idx, :][get_parameter_count(self.config):].values
        
        # test forward simulation
        input_forward = tf.reshape(
            tf.convert_to_tensor(self._normalize(input_structure,self.mean,self.std)),
            (1, get_parameter_count(self.config))
        )
        output_forward = self._denormalize(
            self.model_forward(input_forward),
            mean=self.mean_y,std=self.std_y
        )
        
        # test inverse design
        output_structure, output_spectrum = self.design(input_spectrum)
        
        # calculate error 
        result = {"forward":
                  {"mse": tf.keras.losses.mean_squared_error(
                    self._normalize(input_spectrum,self.mean_y,self.std_y),
                    self.model_forward(input_forward)
                    )[0].numpy()/2,
                   "true":(self.ks, input_spectrum),
                   "pred":(self.ks, output_forward)
                  },
                  "inverse":
                  {"mse":tf.keras.losses.mean_squared_error(
                        self._normalize(input_spectrum,self.mean_y,self.std_y),
                        tf.reshape(
                            self._normalize(np.array(output_spectrum),self.mean_y,self.std_y),
                            (1, self.config["spectral_resolution"])
                        )
                        )[0].numpy()/2,
                   "true":(self.ks, input_spectrum),
                   "pred":(self.ks, output_spectrum)
                  }
                 }
        
        if plot:
            import matplotlib.pyplot as plt
            print("Forward prediction")
            print(f"Input structure : {input_structure}")
            print(f"MSE: {result['forward']['mse']}")
            plt.plot(self.ks, input_spectrum, self.ks, output_forward)
            plt.show()

            print("Inverse design")
            print(f"Possible structure : {[round(x * 1e9, 1) for x in input_structure]}")
            print(f"Output structure : {[round(x * 1e9, 1) for x in output_structure['d'][1:-1]]}")
            print(f"MSE: {result['inverse']['mse']}")
            plt.plot(self.ks, input_spectrum, self.ks, output_spectrum)
            plt.show()
        
        return result

    def save_model(self,comment=""):
        filename = hashlib.md5(str(self.config).encode()).hexdigest()
        self.model_forward.save(f'models/model_foward_{filename}_{comment}')
        self.model_backward.save(f'models/model_backward_{filename}_{comment}')
        self.model_inverse.save(f'models/model_inverse_{filename}_{comment}')
