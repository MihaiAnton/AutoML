from pandas import DataFrame

from ..abstractModel import AbstractModel
from torch import nn
from ....Exceptions.learnerException import DeepLearningModelException

class DeepLearningModel(AbstractModel):

    POSSIBLE_ACTIVATIONS = ["relu", "linear", "sigmoid"]    #TODO complete with other activations

    def __init__(self, in_size, out_size, config:dict=None):
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


        #create a neural network, named model
        model = self.create_model()



    def predict(self, X: DataFrame) -> DataFrame:
        pass

    def train(self, X: DataFrame, Y:DataFrame, time:int=600):
        pass




    def create_model(self):
        """
            Cretaes a neural network as specified in the configuration
        :return:
        """

        #define network detail
        #get the configured items
        hidden_layers_requested = self._config.get("HIDDEN_LAYERS","smooth")
        activation_requested = self._config.get("ACTIVATIONS", "relu")
        dropout_requested = self._config.get("DROPOUT",0)
        input_layer_size = self._input_count
        output_layer_size = self._output_count

        ##### parse the arguments so they can be used in the network

        ### layers
        hidden_layers = []
        if hidden_layers_requested == "smooth":
            #create a list of hidden layer sizes, always layer i's size being the (i-1) layer's size divide by 2,
            #until the division is less than the output layer
            crt_size = self._input_count//2

            while crt_size > self._output_count:
                hidden_layers.append(crt_size)
                crt_size = crt_size // 2

        else:
            #TODO - add custom layer sizes list
            pass

        ### activations
        if type(activation_requested) not in [str, list]:
            #TODO show warning - activation type provided not understood
            activation_requested = "relu"

        if type(activation_requested) is str:
            if activation_requested in DeepLearningModel.POSSIBLE_ACTIVATIONS:
                activations = activation_requested * (len(hidden_layers) + 1)
            else:
                raise DeepLearningModelException("Not able to use activation function {}".format(activation_requested))

        elif type(activation_requested) is list:
            #the model needs len(hidden_layers) + 1(for the output) activations
            #   - if the list is of this length -> use it
            #   - otherwise, complete with the last element until the end
            if len(activation_requested) == 0:
                #TODO show warning - provided an empty list
                activations = ["relu"] * (len(hidden_layers) + 1)
            else:
                for act in activation_requested:
                    if act not in DeepLearningModel.POSSIBLE_ACTIVATIONS:
                        raise DeepLearningModelException("Not able to use activation function {}".format(activation_requested))
                activations = activation_requested + [activation_requested[-1]] * (len(hidden_layers) + 1 - len(activation_requested))

        ### dropout
        if type(dropout_requested) not in [float, list]:
            # TODO show warning - dropout type provided not understood
            dropout_requested = 0.1

        if type(dropout_requested) is float:
            dropouts = [dropout_requested] * (len(hidden_layers)) #one after each hidden layer
        elif type(dropout_requested) is list:
            #TODO show warning if needed: more dropouts than layers
            dropouts = dropout_requested[:(len(hidden_layers))]



        #create the network class
        class Network(nn.Module):
            def __init__(self):
                super().__init__()



                #layers
                #hidden layers linking
                prev_size = input_layer_size
                for i in range(len(hidden_layers)):
                    layer = nn.Linear(prev_size, hidden_layers[i])
                    prev_size = hidden_layers[i]
                    hidden_layers[i] = layer

                hidden_layers.append(nn.Linear(prev_size, output_layer_size))       #the connection to the output
                self.hidden_layers = hidden_layers


                #activations
                for i in range(len(activations)):
                    activation_layer = None
                    if activations[i] == "relu":
                        activation_layer = nn.ReLU()
                    elif activations[i] == "sigmoid":
                        activation_layer = nn.Sigmoid()
                    else:
                        activation_layer = nn.Identity()

                    activations[i] = activation_layer
                self.activations = activations

                #dropout

                for i in range(len(dropouts)):
                    dropouts[i] = nn.Dropout(dropouts[i])
                self.dropouts  = dropouts


            def forward(self,x):
                #for each hidden layer: apply the weighted transformation, activate and dropout
                for i in range(len(self.hidden_layers)):
                    x = self.hidden_layers[i](x)             #transform

                    if i < len(self.activations[i](x)):      #activate
                        x = self.activations[i](x)

                    if i < len(self.dropouts):          #dropout
                        x = self.dropouts[i](x)

                x = self.hidden_layers[-1](x)
                x = self.activations[-1](x)

                return x


        #return an instance
        return Network()





















