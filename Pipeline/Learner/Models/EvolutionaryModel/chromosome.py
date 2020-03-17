from pandas import DataFrame

from ..abstractModel import AbstractModel


class Chromosome:
    """
        The individual unit from a population.
        It's main task is to hold a model and evaluate it.

        Methods:
            - eval(): evaluates the model's performance on a dataset
            - get_fitness(): returns the fitness/score of the model
            - get_model(): returns the model within the chromosome
    """

    def __init__(self, model: AbstractModel):
        """
            Initializes a chromosome with a model
        :param model: the model that the chromosome operates on
        """
        self._genotype = model
        self._phenotype = None

    def eval(self, X: DataFrame, Y: DataFrame) -> float:
        """
            Evaluates the model and returns a score (the fitness of the chromosome).
        :param X: the data to predict an output from
        :param Y: the data to compare the output to
        :return: chromosome's fitness
        """
        # evaluate the model
        # TODO
        # save the evaluation as phenotype

        # return the value

    def get_fitness(self) -> float:
        """
            Returns the model evaluation score
        :return: the phenotype
        """
        return self._phenotype

    def get_model(self) -> AbstractModel:
        """
            Returns the model
        :return: the model within the chromosome
        """
        return self._genotype
