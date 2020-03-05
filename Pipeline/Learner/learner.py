from ..Mapper import Mapper
from pandas import DataFrame
from .Models.abstractModel import AbstractModel
from .Models.modelFactory import ModelFactory

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
                        - expected to get the TRAINING_CONFIG section of the config file
        """

        self._config = config
        self._mapper = Mapper('Learner')
        self._model_factory = ModelFactory(self._config)


    def learn(self, X:DataFrame, Y:DataFrame, input_size:None, output_size:None)->AbstractModel:
        """
            Learns based on the configuration provided.
        :return: learnt model and statistics
        """
        #parameter validation
        #TODO


        #input and output size
        if input_size is None:
            input_size = X.shape[1]
        if output_size is None:
            output_size  = Y.shape[1]

        self._mapper.set("input_size", input_size)
        self._mapper.set("output_size", output_size)

        #creates a model
        model = self._model_factory.create_model(in_size=input_size, out_size=output_size)

        #trains the model
        train_time = self._convert_train_time(self._config.get("TIME","10m"))
        model.train(X,Y, train_time)

        #returns it
        return model

    def _convert_train_time(self, time:str)->int:
        """
            Converts the time from "xd yh zm ts" into seconds
        :param time: string containing the time in textual format -number of days , hours, minutes and seconds
        :return: the time in seconds
        """
        mapping = {}
        crt_count = 0

        for c in time:      #for each character
            if c.isnumeric():
                crt_count = crt_count * 10 + int(c)
            elif c in "dhms":       #days hours minutes seconds
                mapping[c] = mapping.get(c, 0) + crt_count
                crt_count = 0
            else:
                crt_count = 0

        seconds =  mapping.get("s",0) + mapping.get("m", 0)*60 + mapping.get("h",0)*(60*60) + mapping.get("d",0)*24*60*60




    def get_mapper(self)->'Mapper':
        """
            Returns the mapper that contains data about training
        :return: the mapper
        """
        return self._mapper









































