from .abstractModel import AbstractModel
from .SpecializedModels.deepLearningModel import DeepLearningModel
from ...Exceptions.learnerException import ModelSelectionException


class ModelFactory:

    def __init__(self, config: dict = None):
        """
            Inits a model factory, responsible of returning untrained models as specified in the config.
            Based on the configuration provided it will create a new class derived from AbstractModel which will implement
        the model asked for.
            The class is responsible for the aggregation of different deepe learning/ machine learning libraries,
        since it only has to return an AbstractModel class instance, regardless of what framework is used behind
        the train and predict methods.
        """
        if config is None:
            config = {}

        self._config = config

    def create_model(self, in_size, out_size) -> AbstractModel:
        """
            Creates a model as specified in the configuration.
        :return: the model created
        :param in_size: the input size of the model
        :param out_size: the predicted size of the model
        :raise: ModelSelectionException
        """

        # decide which kind of model has to be created and call the right method

        requested_type = self._config.get("TYPE", "default")

        if requested_type == "some_future_type":
            # do specific stuff
            pass
        elif requested_type == "default":  # choose the default model
            return self._create_default_model(in_size=in_size, out_size=out_size)

        else:
            raise ModelSelectionException("could not create model of type {}".format(requested_type))

    # ------------------------------------------------------  TYPE: default   ------------------------------------------------------

    def _create_default_model(self, in_size, out_size):
        """
            Creates a default model which is expected to receive in_size variables and predict out_size variables
        :param in_size: the size of the imput
        :param out_size: the size of the predicted output
        :return: model instance
        """

        default_model_type = self._config.get("DEFAULT_MODEL", "neural_network")

        if default_model_type == "neural_network":
            model = DeepLearningModel(in_size, out_size, config=self._config.get("DEFAULT_MODEL_CONFIG", {}))

        else:  # TODO add other methods
            pass

        return model
    #######################################################  TYPE: default   #######################################################

    # ------------------------------------------------------  TYPE: evolutionary   ------------------------------------------------------
    # TODO later
    #######################################################  TYPE: evolutionary   #######################################################
