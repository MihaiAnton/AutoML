import json
import os
import pickle
import binascii

from pandas import DataFrame

from ..abstractModel import AbstractModel
from torch import nn, optim, tensor
from ....Exceptions.learnerException import DeepLearningModelException
from sklearn.model_selection import train_test_split
from random import randrange
import time
import numpy as np
import pandas as pd
from torch import save as torch_save
from torch import load as torch_load

from .modelTypes import DEEP_LEARNING_MODEL as MODEL_TYPE


class ModuleList(object):
    """
        Pytorch implementation of dynamic attribute list
    """

    # Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if self.num_module > 0 and i == -1:
            i = self.num_module - 1

        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class DeepLearningModel(AbstractModel):
    POSSIBLE_ACTIVATIONS = ["relu", "linear", "sigmoid"]  # TODO complete with other activations
    DEFAULT_ACTIVATION = "linear"
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LAYER_SIZE = 64
    DEFAULT_DROPOUT = 0.1
    DEFAULT_OPTIMIZER = "SGD"
    DEFAULT_LR = 0.01
    DEFAULT_MOMENTUM = 0.4
    TEMPORARY_FILE = ".tmp_model_file"

    def __init__(self, in_size, out_size, config: dict = None, predicted_name: list = None, dictionary=None):
        """
            Initializes a deep learning model.
            :param in_size: the input size of the neural network
            :param out_size: the predicted size of the network
            :param config: the configuration map
        """
        if type(dictionary) is dict:  # for internal use;
            self._init_from_dictionary(dictionary)  # load from a dictionary when loading from file the model
            return

        if config is None:
            config = {}

        self._predicted_name = predicted_name
        if predicted_name is None:
            self._predicted_name = ["predicted_{}".format(i) for i in range(out_size)]

        self._config = config
        self._input_count = in_size
        self._output_count = out_size

        # create a neural network, named model
        self._model = self.create_model()
        self._optimizer = None
        self._train_mode = True

    def predict(self, X: DataFrame) -> DataFrame:
        """
            Predicts a set of data transformed to fit to the model's input expectation
        :param X: dataset to predict
        :return: DataFrame with the output
        """
        if self._train_mode:
            self._model.eval()
            self._train_mode = False

        processed = tensor(X.to_numpy()).float()
        output = self._model(processed)

        numpy_array = np.asarray(output.detach())

        df = pd.DataFrame(numpy_array, columns=self._predicted_name)
        return df

    def train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, validation_split: float = None):
        """
            Trains the model according to the specifications provided.
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param X: the dependent variables to train with
        :param Y: the predicted variables
        :param train_time: the training time in seconds, default 10 minutes
        :return: self (trained model)
        """
        # define an optimizer
        # should be defined in configuration - a default one will be used now for the demo
        if not self._train_mode:
            self._train_mode = True
            self._model.train()

        criterion = nn.BCELoss()

        requested_optimizer = self._config.get("OPTIMIZER", self.DEFAULT_OPTIMIZER)
        requested_lr = self._config.get("LEARNING_RATE", self.DEFAULT_LR)
        requested_momentum = self._config.get("MOMENTUM", self.DEFAULT_MOMENTUM)

        params = self._model.parameters()
        if requested_optimizer == "SGD":
            optimizer = optim.SGD(params, lr=requested_lr, momentum=requested_momentum)
        elif requested_optimizer == "Adam":
            optimizer = optim.Adam(params, lr=requested_lr)
        else:
            raise DeepLearningModelException("Optimizer {} not understood.".format(requested_optimizer))

        self._optimizer = optimizer
        batch_size = self._config.get("BATCH_SIZE", self.DEFAULT_BATCH_SIZE)

        # create the train and validation datasets
        if validation_split is None:
            x_train = tensor(X.to_numpy()).float()
            y_train = tensor(Y.to_numpy()).float()
        else:
            if type(validation_split) != float:
                raise DeepLearningModelException("Parameter validation_split should be None or float in range [0,1)")
            if validation_split < 0 or validation_split >= 1:
                validation_split = 0.2
                # TODO warning - validation is out of limits, using default value 0.2

            x_train, x_val, y_train, y_val = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=validation_split,
                                                              random_state=randrange(2048))

            x_train = tensor(x_train).float()
            x_val = tensor(x_val).float()
            y_train = tensor(y_train).float()
            y_val = tensor(y_val).float()

        # prepare for time handling
        seconds_count = 0
        epochs = 0

        start_time = time.time()
        requested_finish = start_time + train_time
        expected_finish = requested_finish

        keep_training = True

        # train the model - handle time somehow
        while keep_training:
            keep_training = False

            epoch_start = time.time()

            running_loss = 0
            start_index = 0

            while start_index < x_train.shape[0]:
                batch_x = x_train[start_index:start_index + batch_size]
                batch_y = y_train[start_index:start_index + batch_size]

                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                start_index += batch_size

            else:
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                # predict the end of the training session
                seconds_count += epoch_duration
                epochs += 1

                time_per_epoch = seconds_count / epochs

                epochs_to_complete = (requested_finish - epoch_end) / time_per_epoch  # how much time available split to
                # the average epoch time
                if epochs_to_complete - int(epochs_to_complete) >= 0.5:
                    epochs_to_complete = int(epochs_to_complete) + 1
                else:
                    epochs_to_complete = int(epochs_to_complete)

                if epochs_to_complete > 0:
                    keep_training = True

                if epochs % 100 == 99:

                    # print("--",time.localtime(epoch_end), epochs_to_complete, time_per_epoch)
                    expected_finish = epoch_end + epochs_to_complete * time_per_epoch

                    # printed format
                    date = time.localtime(expected_finish)
                    if time.localtime(epoch_end).tm_mday == date.tm_mday:
                        day = ""
                    elif time.localtime(epoch_end).tm_mday == date.tm_mday - 1:
                        day = "tomorrow|"
                    else:
                        day = "{}/{}/{}|".format(date.tm_mday, date.tm_mon, date.tm_year)

                    hour = date.tm_hour
                    minute = date.tm_min
                    second = date.tm_sec

                    if not (validation_split is None):
                        pred_val = self._model(x_val)
                        loss_val = criterion(pred_val, y_val).item()

                        print("Epoch {} - Training loss: {} - Validation loss: {} - ETA: {}{}:{}:{}"
                              .format(epochs, running_loss / x_train.shape[0], loss_val / x_val.shape[0],
                                      day, hour, minute, second))
                    else:
                        print("Epoch {} - Training loss: {} - ETA: {}{}:{}:{}".format(epochs,
                                                                                      running_loss / x_train.shape[0],
                                                                                      day, hour, minute, second))

        return self

    def create_model(self):
        """
            Cretaes a neural network as specified in the configuration
        :return:
        """

        # define network detail
        # get the configured items
        hidden_layers_requested = self._config.get("HIDDEN_LAYERS", "smooth")
        activation_requested = self._config.get("ACTIVATIONS", self.DEFAULT_ACTIVATION)
        dropout_requested = self._config.get("DROPOUT", self.DEFAULT_DROPOUT)
        input_layer_size = self._input_count
        output_layer_size = self._output_count

        ##### parse the arguments so they can be used in the network

        ### layers
        hidden_layers = []
        if hidden_layers_requested == "smooth":
            # create a list of hidden layer sizes, always layer i's size being the (i-1) layer's size divide by 2,
            # until the division is less than the output layer
            crt_size = self._input_count // 2

            while crt_size > self._output_count:
                hidden_layers.append(crt_size)
                crt_size = crt_size // 2

        else:
            for layer_size in hidden_layers_requested:
                if layer_size == 0:
                    layer_size = self.DEFAULT_LAYER_SIZE
                    # TODO:show warning, empty hidden layer
                if layer_size < 0:
                    layer_size = -layer_size
                hidden_layers.append(layer_size)
            pass

        ### activations
        if type(activation_requested) not in [str, list]:
            # TODO show warning - activation type provided not understood
            activation_requested = self.DEFAULT_ACTIVATION

        if type(activation_requested) is str:
            if activation_requested in DeepLearningModel.POSSIBLE_ACTIVATIONS:
                activations = [activation_requested] * (len(hidden_layers) + 1)
            else:
                raise DeepLearningModelException("Not able to use activation function {}".format(activation_requested))

        elif type(activation_requested) is list:
            # the model needs len(hidden_layers) + 1(for the output) activations
            #   - if the list is of this length -> use it
            #   - otherwise, complete with the last element until the end
            if len(activation_requested) == 0:
                # TODO show warning - provided an empty list
                activations = [self.DEFAULT_ACTIVATION] * (len(hidden_layers) + 1)
            else:
                for act in activation_requested:
                    if act not in DeepLearningModel.POSSIBLE_ACTIVATIONS:
                        raise DeepLearningModelException(
                            "Not able to use activation function {}".format(activation_requested))
                activations = activation_requested + [activation_requested[-1]] * (
                        len(hidden_layers) + 1 - len(activation_requested))

        ### dropout
        if type(dropout_requested) not in [float, list]:
            # TODO show warning - dropout type provided not understood
            dropout_requested = self.DEFAULT_DROPOUT

        if type(dropout_requested) is float:
            dropouts = [dropout_requested] * (len(hidden_layers))  # one after each hidden layer
        elif type(dropout_requested) is list:
            # TODO show warning if needed: more dropouts than layers
            dropouts = dropout_requested[:(len(hidden_layers))]

        # create the network class
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

                # self._l1 = nn.Linear(input_layer_size, output_layer_size)
                # self._a1 = nn.Sigmoid()

                # define attribute lists
                self._layers = ModuleList(self, "layer_")
                self._layer_count = 0

                self._activations = ModuleList(self, "activation_")
                self._activations_count = 0

                self._dropouts = ModuleList(self, "dropout_")
                self._dropout_count = 0

                # layers
                # hidden layers linking
                prev_size = input_layer_size
                for i in range(len(hidden_layers)):
                    layer = nn.Linear(prev_size, hidden_layers[i])
                    prev_size = hidden_layers[i]
                    self._layers.append(layer)
                    self._layer_count += 1

                self._layers.append(nn.Linear(prev_size, output_layer_size))  # the connection to the output
                self._layer_count += 1

                # activations
                for i in range(len(activations)):
                    activation_layer = None
                    if activations[i] == "relu":
                        activation_layer = nn.ReLU()
                    elif activations[i] == "sigmoid":
                        activation_layer = nn.Sigmoid()
                    else:
                        activation_layer = nn.Sigmoid()  # TODO modify in identiyy

                    self._activations.append(activation_layer)
                    self._activations_count += 1

                # dropout
                for i in range(len(dropouts)):
                    self._dropouts.append(nn.Dropout(dropouts[i]))
                    self._dropout_count += 1

            def forward(self, x):
                # for each hidden layer: apply the weighted transformation, activate and dropout
                for i in range(self._layer_count - 1):
                    x = self._layers[i](x)  # transform

                    if i < len(self._activations):  # activate
                        x = self._activations[i](x)

                    if i < len(self._dropouts):  # dropout
                        x = self._dropouts[i](x)

                x = self._layers[-1](x)
                x = self._activations[-1](x)

                return x

        # return an instance
        net = Network()
        return net

    def to_dict(self) -> dict:
        """
            Returns a dictionary representation of the model for further file saving.
        :return: dictionary with model encoding
        """

        # get the model weights
        model_binary = ""

        model = pickle.dumps(self._model.state_dict())

        # torch_save(self._model.state_dict(), self.TEMPORARY_FILE)
        # # create a temporary file with the binaries of the model
        # with open(self.TEMPORARY_FILE, 'rb') as tmp_file:  # include the binaries in a new created json file
        #     model_binary = tmp_file.read()
        #     model_binary = json.dumps(model_binary)



        data = {
            "MODEL": model,
            "METADATA": {
                "PREDICTED_NAME": self._predicted_name,
                "CONFIG": self._config,
                "INPUT_COUNT": self._input_count,
                "OUTPUT_COUNT": self._output_count,
            }
        }

        return data

    @staticmethod
    def load(source):
        """
            Loads the DeepLearningModel from a source that can be a file with a json or an already
        parsed dictionary.
        :param source: str(with previously saved model) or dict(with the dictionary previously
                        returned by to_dict)
        :return: model
        """
        if type(source) is str:
            try:
                with open(source) as f:
                    dictionary = json.load(f)
                    model = DeepLearningModel(0, 0, dictionary=dictionary)
                return model
            except Exception as e:
                raise Exception("Could not init deep learning model from file {}.".format(source))

        elif type(source) is dict:
            model = DeepLearningModel(0, 0, dictionary=source)
            return model
        else:
            raise DeepLearningModelException(
                "Could not load deep learning model from file or dict of type {}".format(type(source)))

    def model_type(self) -> str:
        """
            Returns the model type; in this case -> DEEP_LEARNING_MODEL
        :return:
        """
        return MODEL_TYPE

    def _init_from_dictionary(self, d: dict):
        """
            Inits the model from dictionary; sets the attributes to be as they were before saving.
            It is assumed theat the dictionary provided here is the one intended for this model type.
                - should only be called from the constructor

        :param d: dictionary previously created by to_dict
        :return: None
        """

        data = d.get("METADATA")
        model = d.get("MODEL")

        # init the data
        self._predicted_name = data.get("PREDICTED_NAME")
        self._config = data.get("CONFIG")
        self._input_count = data.get("INPUT_COUNT")
        self._output_count = data.get("OUTPUT_COUNT")

        # init the model
        self._model = self.create_model()

        # restore the weights
        model_saved = pickle.loads(model)
        self._model.load_state_dict(model_saved)

        self._train_mode = False
        self._model.eval()

    def reset(self):
        sd = self._model.state_dict()

        # self._model = self.create_model()
        self._model.load_state_dict(sd)