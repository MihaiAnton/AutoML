"""
    This file contains all the XO (crossover) methods for chromosomes (models)
"""
from random import choice, random, randint

from ..SpecializedModels import *


def deep_learning_XO_deep_learning(model1: DeepLearningModel, model2: DeepLearningModel,
                                   in_size: int, out_size: int, task:str):
    """
        Performs crossover between 2 deep learning models.
        For each possible parameter, it performs a combination of the parameters of the parents with a
    probability of 60%. Otherwise, it takes the parameter of one random parent.
    :param task: the task carried out by the model ("REGRESSION" / "CLASSIFICATION" )
    :param out_size: the output of the model
    :param in_size: the input of the model
    :param model1: deep learning model
    :param model2: deep learning model
    :return: deep learning model
    """

    # TODO add a better method than just random selection between parents

    config1 = model1.get_config()
    config2 = model2.get_config()
    XO_PROBAB = 0.6

    # optimizer
    optimizer = choice([config1.get("OPTIMIZER"), config2.get("OPTIMIZER")])

    # learning rate
    if random() <= XO_PROBAB:
        learning_rate = .5 * (config1.get("LEARNING_RATE") + config2.get("LEARNING_RATE"))
    else:
        learning_rate = choice([config1.get("LEARNING_RATE"), config2.get("LEARNING_RATE")])

    # momentum
    if random() <= XO_PROBAB:
        momentum = .5 * (config1.get("MOMENTUM") + config2.get("MOMENTUM"))
    else:
        momentum = choice([config1.get("MOMENTUM"), config2.get("MOMENTUM")])

    # hidden_layers
    layers = choice([config1.get("HIDDEN_LAYERS"), config2.get("HIDDEN_LAYERS")])

    # if random() > XO_PROBAB or type(config1.get("HIDDEN_LAYERS")) != type(config2.get("HIDDEN_LAYERS")):
    #     layers = choice([config1.get("HIDDEN_LAYERS"), config2.get("HIDDEN_LAYERS")])
    # else:
    #     if type(config1.get("HIDDEN_LAYERS")) is str:
    #         layers = config1.get("HIDDEN_LAYERS")
    #     else:
    #         layers = []
    # TODO find another method for combining hidden layers

    # activations
    if random() > XO_PROBAB or type(config1.get("ACTIVATIONS")) != type(config2.get("ACTIVATIONS")):
        activation = choice([config1.get("ACTIVATIONS"), config2.get("ACTIVATIONS")])
    else:
        if type(config1.get("ACTIVATIONS")) is str:
            activation = choice([config1.get("ACTIVATIONS"), config2.get("ACTIVATIONS")])
        else:
            activation = choice([config1.get("ACTIVATIONS"), config2.get("ACTIVATIONS")])
            # TODO find a better method

    # dropout
    if random() <= XO_PROBAB:
        dropout = .5 * (config1.get("DROPOUT") + config2.get("DROPOUT"))
    else:
        dropout = choice([config1.get("DROPOUT"), config2.get("DROPOUT")])

    offspring_config = {
        "CRITERION": config1.get("CRITERION", "undefined"),
        "OPTIMIZER": optimizer,
        "LEARNING_RATE": learning_rate,
        "MOMENTUM": momentum,
        "HIDDEN_LAYERS": layers,
        "ACTIVATIONS": activation,
        "DROPOUT": dropout
    }

    return DeepLearningModel(in_size, out_size, task, offspring_config)
