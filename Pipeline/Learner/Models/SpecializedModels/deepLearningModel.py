from pandas import DataFrame

from ..abstractModel import AbstractModel
from torch import nn, optim, tensor
from ....Exceptions.learnerException import DeepLearningModelException


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

    def __init__(self, in_size, out_size, config: dict = None):
        """
            Initializes a deep learning model.
            :param in_size: the input size of the neural network
            :param out_size: the predicted size of the network
            :param config: the configuration map
        """
        if config is None:
            config = {}

        self._config = config
        self._input_count = in_size
        self._output_count = out_size

        # create a neural network, named model
        self._model = self.create_model()

    def predict(self, X: DataFrame) -> DataFrame:
        pass

    def train(self, X: DataFrame, Y: DataFrame, time: int = 600):
        """
            Trains the model according to the specifications provided.
        :param X: the dependent variables to train with
        :param Y: the predicted variables
        :param time: the training time
        :return: self (trained model)
        """
        # define an optimizer
        # should be defined in configuration - a default one will be used now for the demo
        criterion = nn.BCELoss()
        params = self._model.parameters()
        optimizer = optim.SGD(params, lr=0.01, momentum=0.9)

        BATCH_SIZE = 64
        EPOCHS = 12000

        # convert the data to tensors
        X = tensor(X.to_numpy()).float()
        Y = tensor(Y.to_numpy()).float()

        # train the model - handle time somehow
        for e in range(EPOCHS):
            running_loss = 0
            start_index = 0

            while start_index < X.shape[0]:
                batchX = X[start_index:start_index + BATCH_SIZE]
                batchY = Y[start_index:start_index + BATCH_SIZE]

                optimizer.zero_grad()
                output = self._model(batchX)
                loss = criterion(output, batchY)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                start_index += BATCH_SIZE

            else:
                if e % 100 == 99:
                    print("Epoch {} - Training loss: {}".format(e, running_loss / X.shape[0]))

    def create_model(self):
        """
            Cretaes a neural network as specified in the configuration
        :return:
        """

        # define network detail
        # get the configured items
        hidden_layers_requested = self._config.get("HIDDEN_LAYERS", "smooth")
        activation_requested = self._config.get("ACTIVATIONS", self.DEFAULT_ACTIVATION)
        dropout_requested = self._config.get("DROPOUT", 0)
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
            # TODO - add custom layer sizes list
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
            dropout_requested = 0.1

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
                for i in range(self._layer_count-1):
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
