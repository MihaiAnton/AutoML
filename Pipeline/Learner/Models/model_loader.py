import json
import pickle

from .SpecializedModels.modelTypes import *
from .SpecializedModels.deepLearningModel import DeepLearningModel
from ...Exceptions.learnerException import ModelLoaderException


def load_model(source):
    """
        Loads a generic model from file. After successful loading the model is ready to be used
    for training or prediction.
        Contains all the logic for loading all model types. Requires files that have 2 keys:
            - MODEL_TYPE
            - MODEL_DATA
            ;which must be returned in each instance of to_dict of any AbstractModel implementation

    :param source: str(file with saved dictionary) or dictionary(with model)
    :return: model instance
    """
    if type(source) is str:
        with open(source, 'rb') as f:
            dictionary = pickle.load(f)

    elif type(source) is dict:
        dictionary = source

    else:
        raise ModelLoaderException(
            "Could not load model from data type {}".format(type(source)))

    model = None
    model_type = dictionary.get("MODEL_TYPE", "undefined")
    model_data = dictionary.get("MODEL_DATA")

    if model_type == DEEP_LEARNING_MODEL:
        model = DeepLearningModel(0, 0, dictionary=model_data)
    elif model_type == "":  # TODO add models as they are added in the SpecializedModels module
        pass
    else:
        raise ModelLoaderException(
            "Could not load model of type {}".format(model_type))

    return model