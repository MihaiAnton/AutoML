from pandas import DataFrame

from .abstractLearner import AbstractLearner

class DefaultLearner(AbstractLearner):

    def __init__(self, config:dict={}):
        super().__init__(mapper_name="DefaultLearner")

        self._config = config




    def learn(self, X: DataFrame) -> DataFrame:
        pass