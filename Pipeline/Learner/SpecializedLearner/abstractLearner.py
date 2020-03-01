from abc import ABC, abstractmethod
from pandas import DataFrame
from ...Mapper.mapper import Mapper


class AbstractLearner(ABC):

    def __init__(self, mapper_name:str="SpecializedLearner"):
        """
            Inits an abstract learner.
            Should have a mapper in order to record changes.
        """
        self._mapper = Mapper(mapper_name)

    @abstractmethod
    def learn(self, X:DataFrame, Y:DataFrame):
        """
            Learns based on the (X,Y) tuple provided and returns a trained model.
        :param Y: predicted variable
        :param X: predictor variables
        :return: the trained model
        """
        pass


    def get_mapper(self)->'Mapper':
        """
            Returns the current mapper.
        :return: mapper
        """
        return self._mapper