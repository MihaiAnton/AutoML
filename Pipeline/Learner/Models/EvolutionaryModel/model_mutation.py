"""
    This file contains methods t=for mutating different types of models
"""

from ..SpecializedModels import DeepLearningModel
from random import choice, random, randrange


def deep_learning_mutation(model1: DeepLearningModel, in_size: int, out_size: int, task: str, choice_config: dict
                           ) -> DeepLearningModel:
    """
        Performs a random mutation on the deep learning model

    :param model1: the model to be mutated
    :param in_size: the input size
    :param out_size: the output size
    :param task: the task to be performed
    :param choice_config: the choices configuration dictionary
    :return: new mutated model
    """
    config = model1.get_config()

    # the optimiser should be left in place

    # learning_rate
    learning_rate = config.get("LEARNING_RATE")
    learning_rate = learning_rate + learning_rate * choice([-2, -1, 1]) * random() * 0.2

    learning_rate = max(0.00001, learning_rate)  # just be sure it does not get negative

    # momentum
    momentum = config.get("MOMENTUM")
    momentum = momentum + momentum * choice([-1, 1]) * random() * 0.2

    momentum = max(0.00001, momentum)  # just be sure it does not get negative

    # hidden layers
    layers = config.get("HIDDEN_LAYERS")
    if type(layers) is list:
        for i in range(len(layers)):
            layer = layers[i] + choice([1, -1]) * layers[i] * random() * 0.2
            layers[i] = int(layer)

    # activations
    activations = config.get("ACTIVATIONS")
    if type(activations) is list:
        position = randrange(0, len(activations))
        activations[position] = choice(choice_config.get("ACTIVATION_CHOICES", []))

    # dropout
    dropout = config.get("DROPOUT")
    if type(dropout) in [int, float]:
        dropout = dropout + dropout * choice([-1, 1]) * random() * 0.2
        dropout = max(0.00001, momentum)  # just be sure it does not get negative
        dropout = min(momentum, 1)         # and no more than 1

    offspring_config = {
        "CRITERION": config.get("CRITERION", "undefined"),
        "OPTIMIZER": config.get("OPTIMIZER"),
        "LEARNING_RATE": learning_rate,
        "MOMENTUM": momentum,
        "HIDDEN_LAYERS": layers,
        "ACTIVATIONS": activations,
        "DROPOUT": dropout
    }

    return DeepLearningModel(in_size, out_size, task, offspring_config)
