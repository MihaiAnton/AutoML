import json

from .SpecializedModels.modelTypes import *
from .SpecializedModels.deepLearningModel import DeepLearningModel
from ...Exceptions.learnerException import ModelLoaderException


def load_model(source):
    """
        Loads a generic model from file.
    :param source: str(file with saved dicitonary) or dictionary(with model)
    :return: model instance
    """
    if type(source) is str:
        with open(source) as f:
            dictionary = json.load(f)

    elif type(source) is dict:
        dictionary = source

    else:
        raise ModelLoaderException(
            "Could not load model from data type {}".format(type(source)))

    model = None
    model_type = dictionary.get("DATA", {}).get("MODEL_TYPE", "undefined")

    if model_type == DEEP_LEARNING_MODEL:
        model = DeepLearningModel(0, 0, dictionary=dictionary)
    elif model_type == "":  # TODO add models as they are added in the SpecializedModels module
        pass
    else:
        raise ModelLoaderException(
            "Could not load model of type {}".format(model_type))

    return model
