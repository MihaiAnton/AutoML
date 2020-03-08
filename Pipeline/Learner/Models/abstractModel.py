from abc import ABC, abstractmethod

from pandas import DataFrame


class AbstractModel(ABC):
    """
        The result of the pipeline.
        It's main task is to predict after a training session has been done.
    """

    @abstractmethod
    def train(self, X: DataFrame, Y: DataFrame, time: int = 600):
        """
            Trains the model with the data provided.
        :param time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :return: the model
        """

    @abstractmethod
    def predict(self, X: DataFrame) -> DataFrame:
        """
            Predicts the output of X based on previous learning
        :param X: DataFrame; the X values to be predicted into some Y Value
        :return: DataFrame with the predicted data
        """

    def __call__(self, X: DataFrame) -> DataFrame:
        """
            Calls the predict method.
        :param X: data to be predicted
        :return: predicted data
        """
        self.predict(X)
