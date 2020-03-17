import pickle
from pandas import DataFrame

from .. import AbstractModel
from ....Exceptions import EvolutionaryModelException
from .. import load_model
from .. import EVOLUTIONARY_MODEL
from .population import Population


class EvolutionaryModel(AbstractModel):
    """
        The evolutionary model is responsible, like any other AbstractModel, to learn from data and eventually predict,
    but it also searches for the best performing model using evolutionary algorithms.

        The API is similar to the AbstractModel's API
    """

    def __init__(self, in_size: int, out_size: int, config: dict = None):
        """
            Initializes a evolutionary model
        :param in_size: the size of the input data
        :param out_size: the size that needs to be predicted
        :param config: the configuration dictionary
        """
        if config is None:
            config = {}

        self._config = config
        self._input_size = in_size
        self._output_size = out_size

        self._population = self._create_population(in_size, out_size,
                                                   config)  # the population for the evolutionary algorithm
        self._model = None  # the final model, after the evolutionary phase

    @staticmethod
    def _create_population(in_size: int, out_size: int, config: dict = {}) -> Population:
        """
            Creates a population as configured in the config file
        :param in_size: the size of the input data
        :param out_size: the size that needs to be predicted
        :param config: the configuration dictionary
        :return: a population of models
        """
        population_size = config.get("POPULATION_SIZE", 10)
        population = Population(in_size, out_size, population_size, config)
        return population

    def train(self, X: DataFrame, Y: DataFrame, time: int = 600, callbacks: list = None) -> 'AbstractModel':
        """
                Trains the model with the data provided.
            :param callbacks: a list of predefined callbacks that get called at every epoch
            :param time: time of the training session in seconds: default 10 minutes
            :param X: the independent variables in form of Pandas DataFrame
            :param Y: the dependents(predicted) values in form of Pandas DataFrame
            :return: the model
        """

        # searches for the best model
        # using epochs now, convert to time later
        EPOCHS = 10

        # initial evaluation
        self._population.eval(X, Y)

        for epoch in range(EPOCHS):
            print("======================= EPOCH {}".format(epoch))
            mother = self._population.selection()  # get the parents
            father = self._population.selection()

            offspring = self._population.XO(mother, father)  # combine them
            offspring = self._population.mutation(offspring)  # perform a mutation

            score = offspring.eval(X, Y)  # evaluate the offspring
            # TODO if self._model is None or the score is better( higher or lower -> decide) replace the old model

            self._population.replace(offspring)  # add it in the population

        # trains the best model
        train_time = 0 #decide the train time
        self._model.train(X,Y, train_time)

        # returns it
        return self._model

    def predict(self, X: DataFrame) -> DataFrame:
        """
                Predicts the output of X based on previous learning
            :param X: DataFrame; the X values to be predicted into some Y Value
            :return: DataFrame with the predicted data
        """
        if self._model is None:
            raise EvolutionaryModelException("Train the model before performing a prediction.")

        return self._model.predict(X)

    def to_dict(self) -> dict:
        """
            Returns a dictionary representation of the model for further file saving.
        :return: dictionary with model encoding


        """
        # !!! should match _init_from_dictionary loading format
        # get the model data
        model = None
        if not (self._model is None):
            model = self._model.to_dict()

        data = {
            "MODEL": model,
            "METADATA": {
                # TODO
            }
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "MODEL_DATA": data
        }

    def _init_from_dictionary(self, d: dict):
        """
            Inits the model from dictionary; sets the attributes to be as they were before saving.
            It is assumed that the dictionary provided here is the one intended for this model type.
                - should only be called from the constructor

        :param d: dictionary previously created by to_dict
        :return: None
        """
        # !!! should match to_dict loading format
        d = d.get("MODEL_DATA")
        data = d.get("METADATA")
        model = d.get("MODEL")

        # init the data
        # TODO

        # init the model
        self._model = load_model(model)

    def model_type(self) -> str:
        return EVOLUTIONARY_MODEL

    def _description_string(self) -> str:
        if self._model is None:
            return "Evolutionary Model - Not configured"
        else:
            # TODO
            return "Evolutionary Model - Best model: {best} | Top models: {top}".format(
                best="TODO",
                top="TODO"
            )