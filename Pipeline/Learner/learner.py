from ..Mapper import Mapper
from pandas import DataFrame
from .model import Model
from .SpecializedLearner.defaultLearner import

class Learner:
    """
        The class that handles the learning inside the pipeline.
        It's main task is to learn from a dataset and return a model.
        Based on a configuration file given as constructor parameter it is able to do a series of tasks like:
            - fit the data on a dataset with a default predefined model (defined in config)
            - fit the data using a series of models and evolutionary algorithms for finding the best one
            - predict the data using a predefined model
    """

    def __init__(self, config:dict={}):
        """
            Creates a learner instance based on the configuration file.
            :param config: dictionary with the configurations for the learning module
        """

        self._config = config
        self._mapper = Mapper('Learner')


        self._learner = self._get_learner()

    def learn(self)->Model:
        """
            Learns based on the configuration provided.
        :return: learnt model and statistics
        """





    def get_mapper(self)->'Mapper':
        """
            Returns the mapper that contains data about training
        :return: the mapper
        """
        return self._mapper

    def _get_learner(self):
        """
            Decides which type of specialized learner to use and returns it.
        :return:
        """
        learner_type = self._config.get("LEARNING_TYPE", "default")

        if learner_type == "default":
            return









































