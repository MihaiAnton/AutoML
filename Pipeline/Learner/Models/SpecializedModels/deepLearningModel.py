import pickle
import warnings

from pandas import DataFrame

from ..abstractModel import AbstractModel
from torch import nn, optim, tensor
from ....Exceptions.learnerException import DeepLearningModelException
from sklearn.model_selection import train_test_split
from random import randrange
import time
import numpy as np
import pandas as pd

from ..modelTypes import DEEP_LEARNING_MODEL
from ..constants import AVAILABLE_TASKS, CLASSIFICATION


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
    """
        AbstractModel implementation using a deep learning actual model.
        The framework used is PyTorch.
    """

    POSSIBLE_ACTIVATIONS = ["relu", "linear", "sigmoid"]  # TODO complete with other activations
    DEFAULT_ACTIVATION = "linear"
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LAYER_SIZE = 64
    DEFAULT_DROPOUT = 0.1
    DEFAULT_OPTIMIZER = "SGD"
    DEFAULT_LR = 0.01
    DEFAULT_MOMENTUM = 0.4
    DEFAULT_CRITERION = "MSE"
    TEMPORARY_FILE = ".tmp_model_file"

    def __init__(self, in_size, out_size, task: str = "", config: dict = None, predicted_name: list = None,
                 dictionary: dict = None):
        """
            Initializes a deep learning model.
            :param in_size: the input size of the neural network
            :param out_size: the predicted size of the network
            :param config: the configuration map (expected to get the NEURAL_NETWORK_CONFIG part of the default model)
            :param task: the type of learning that is wanted to be done
        """
        if type(dictionary) is dict:  # for internal use;
            self._init_from_dictionary(dictionary)  # load from a dictionary when loading from file the model
            return

        # data used for printing
        self._configured = False  # defines if the model has been configured or is still blank
        self._layers = []
        self._activations = []
        self._dropouts = []

        if config is None:
            config = {}

        self._predicted_name = predicted_name

        # configuration parameters
        self._task = task
        self._config = config
        self._input_count = in_size
        self._output_count = out_size
        self._classification_mapping = {}

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
            self._train_mode = False
            self._model.eval()

        processed = tensor(X.to_numpy()).float()
        output = self._model(processed)

        numpy_array = np.asarray(output.detach())

        df = pd.DataFrame(numpy_array, columns=self._predicted_name)
        df.fillna(0, inplace=True)  # TODO find better alternative

        if self._task == CLASSIFICATION:
            mapping = self._classification_mapping["mapping"]
            try:
                df_mapped = self._from_categorical(df, mapping)
            except KeyError:
                print(mapping)  # TODO remove later
                print(9)
            df = df_mapped
        # check if the model is set for categorical purpose

        return df

    def train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, validation_split: float = 0.2,
              callbacks: list = None, verbose: bool = True):
        """
            Trains the model according to the specifications provided.
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param X: the dependent variables to train with
        :param Y: the predicted variables
        :param train_time: the training time in seconds, default 10 minutes
        :param verbose: decides whether or not the model prints intermediary outputs
        :return: self (trained model)
        """
        # define the task
        if self._task not in AVAILABLE_TASKS:
            self._task = self._determine_task_type(Y)

        # define the predicted names
        if self._predicted_name is None:
            self._predicted_name = list(Y.columns)

        # if the task is classification - modify the Y column and create a mapping between actual columns and encodings
        if self._task == CLASSIFICATION:
            self._classification_mapping["mapping"] = self._categorical_mapping(Y)
            self._classification_mapping["previous_out_layers"] = self._output_count
            self._classification_mapping["previous_predicted_names"] = self._predicted_name
            self._predicted_name = list(self._classification_mapping["mapping"].keys())
            self._classification_mapping["actual_out_layers"] = len(
                list(self._classification_mapping.get("mapping", {}).keys()))
            Y = self._to_categorical(Y, self._classification_mapping["mapping"])
            self._output_count = self._classification_mapping["actual_out_layers"]
            self._model = self.create_model()

        if not self._train_mode:
            self._train_mode = True
            self._model.train()

        # define an optimizer
        # should be defined in configuration - a default one will be used now for the demo
        requested_criterion = self._config.get("CRITERION", self.DEFAULT_CRITERION)

        if requested_criterion == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif requested_criterion == "LogLikelihood":
            criterion = nn.NLLLoss()
        elif requested_criterion == "PoissonLogLikelihood":
            criterion = nn.PoissonNLLLoss()
        elif requested_criterion == "BCE":
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()

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

        # create the train and validation data sets
        if validation_split is None:
            x_train = tensor(X.to_numpy()).float()
            y_train = tensor(Y.to_numpy()).float()

            print("Training on {} samples...".format(len(y_train))) if verbose else None
        else:

            if type(validation_split) != float:
                raise DeepLearningModelException("Parameter validation_split should be None or float in range [0,1)")
            if validation_split < 0 or validation_split >= 1:
                validation_split = 0.2
                warnings.warn("DeepLearningModelModel: configured validation percentage is out of bounds; using "
                              "default value 0.2", RuntimeWarning)

            x_train, x_val, y_train, y_val = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=validation_split,
                                                              random_state=randrange(2048))

            x_train = tensor(x_train).float()
            x_val = tensor(x_val).float()
            y_train = tensor(y_train).float()
            y_val = tensor(y_val).float()

            print("Training on {} samples. Validating on {}...".format(len(y_train), len(y_val))) if verbose else None

        # prepare for time handling
        seconds_count = 0
        epochs = 0

        start_time = time.time()
        requested_finish = start_time + train_time

        keep_training = True

        # train the model
        self._model.zero_grad()

        while keep_training:
            keep_training = False

            epoch_start = time.time()

            running_loss = 0
            start_index = 0
            batch_count = 0

            while start_index < x_train.shape[0]:
                batch_count += 1
                batch_x = x_train[start_index:start_index + batch_size]
                batch_y = y_train[start_index:start_index + batch_size]

                optimizer.zero_grad()
                output = self._model(batch_x)

                losses = []
                for out in range(len(self._predicted_name)):
                    losses.append(criterion(output[:, out], batch_y[:, out]))

                loss = losses[0]
                for i in range(1, len(losses)):
                    loss = loss + losses[i]

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

                        self._model.zero_grad()
                        self._model.eval()
                        pred_val = self._model(x_val)
                        loss_val = criterion(pred_val, y_val).item()

                        self._model.train()

                        # previous loss display - not consistent when training loss was compared to validation loss
                        # print("Epoch {} - Training loss: {} - Validation loss: {} - ETA: {}{}:{}:{}"
                        #       .format(epochs, running_loss / x_train.shape[0], loss_val / x_val.shape[0],
                        #               day, hour, minute, second)) if verbose else None

                        print("Epoch {} - Training loss: {} - Validation loss: {} - ETA: {}{}:{}:{}"
                              .format(epochs, running_loss/batch_count, loss_val,
                                      day, hour, minute, second)) if verbose else None

                    else:
                        print("Epoch {} - Training loss: {} - ETA: {}{}:{}:{}".format(epochs,
                                                                                      running_loss / x_train.shape[0],
                                                                                      day, hour, minute,
                                                                                      second)) if verbose else None

        return self

    def create_model(self):
        """
            Creates a neural network as specified in the configuration
        :return:
        """
        self._configured = True
        # define network detail
        # get the configured items
        hidden_layers_requested = self._config.get("HIDDEN_LAYERS", "smooth")
        activation_requested = self._config.get("ACTIVATIONS", self.DEFAULT_ACTIVATION)
        dropout_requested = self._config.get("DROPOUT", self.DEFAULT_DROPOUT)
        input_layer_size = self._input_count
        output_layer_size = self._output_count

        # parse the arguments so they can be used in the network

        # layers
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
                    warnings.warn(
                        "DeepLearningModel: provided layer cannot be empty; using {} as default layer size."
                            .format(self.DEFAULT_LAYER_SIZE), RuntimeWarning)

                if layer_size < 0:
                    layer_size = -layer_size
                hidden_layers.append(layer_size)
            pass
        self._layers = [self._input_count] + hidden_layers + [self._output_count]

        # activations
        if type(activation_requested) not in [str, list]:
            warnings.warn(
                "DeepLearningModel: provided activation {} not understood; using {} as default activation."
                    .format(activation_requested, self.DEFAULT_ACTIVATION), RuntimeWarning)
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
                warnings.warn(
                    "DeepLearningModel: provided empty activations list; using list of default {} activation."
                        .format(self.DEFAULT_ACTIVATION), RuntimeWarning)
                activations = [self.DEFAULT_ACTIVATION] * (len(hidden_layers) + 1)
            else:
                for act in activation_requested:
                    if act not in DeepLearningModel.POSSIBLE_ACTIVATIONS:
                        raise DeepLearningModelException(
                            "Not able to use activation function {}".format(activation_requested))
                activations = activation_requested + [activation_requested[-1]] * (
                        len(hidden_layers) + 1 - len(activation_requested))
        self._activations = activations

        # dropout
        if type(dropout_requested) not in [float, list, int]:
            warnings.warn(
                "DeepLearningModel: provided dropout type {} not understood; using {} as default dropout."
                    .format(type(dropout_requested),self.DEFAULT_DROPOUT), RuntimeWarning)
            dropout_requested = self.DEFAULT_DROPOUT

        if type(dropout_requested) in [float, int]:
            dropouts = [dropout_requested] * (len(hidden_layers))  # one after each hidden layer
        elif type(dropout_requested) is list:
            dropouts = dropout_requested[:(len(hidden_layers))]
        self._dropouts = dropouts

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
                        activation_layer = nn.Identity()

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

    def model_type(self) -> str:
        """
            Returns the model type; in this case -> DEEP_LEARNING_MODEL
        :return:
        """
        return DEEP_LEARNING_MODEL

    def to_dict(self) -> dict:
        """
            Returns a dictionary representation of the model for further file saving.
        :return: dictionary with model encoding


        """
        # !!! should match _init_from_dictionary loading format
        # get the model data
        model = pickle.dumps(self._model.state_dict())

        data = {
            "MODEL": model,
            "METADATA": {
                "PREDICTED_NAME": self._predicted_name,
                "CONFIG": self._config,
                "INPUT_COUNT": self._input_count,
                "OUTPUT_COUNT": self._output_count,
                "TASK": self._task,
                "CLASSIFICATION_MAPPING": self._classification_mapping,
                "CONFIGURED": self._configured,
                "LAYERS": self._layers,
                "ACTIVATIONS": self._activations,
                "DROPOUTS": self._dropouts
            }
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "MODEL_DATA": data
        }

    def _init_from_dictionary(self, d: dict):
        """
            Inits the model from dictionary; sets the attributes to be as they were before saving.
            It is assumed that the dictionary provided here is the one intended for this model type.
                - should only be called from the constructor

        :param d: dictionary previously created by to_dict
        :return: None
        """
        # !!! should match to_dict loading format
        d = d.get("MODEL_DATA")
        data = d.get("METADATA")
        model = d.get("MODEL")

        # init the data
        self._predicted_name = data.get("PREDICTED_NAME")
        self._config = data.get("CONFIG")
        self._input_count = data.get("INPUT_COUNT")
        self._output_count = data.get("OUTPUT_COUNT")
        self._task = data.get("TASK")
        self._classification_mapping = data.get("CLASSIFICATION_MAPPING")
        self._configured = data.get("CONFIGURED")
        self._layers = data.get("LAYERS")
        self._activations = data.get("ACTIVATIONS")
        self._dropouts = data.get("DROPOUTS")

        # init the model
        self._model = self.create_model()

        # restore the weights
        model_saved = pickle.loads(model)
        self._model.load_state_dict(model_saved)

        self._train_mode = False
        self._model.eval()

    def _description_string(self) -> str:
        if self._configured is False:
            return "NeuralNetwork - Not configured"
        else:
            return "NeuralNetwork - Layers: {layers} | Activations: {activations} | Dropouts: {dropouts}".format(
                layers=self._layers,
                activations=self._activations,
                dropouts=self._dropouts
            )

    def get_config(self) -> dict:
        return self._config
