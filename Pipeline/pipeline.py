import json
import os
import time
from pandas import DataFrame

from Pipeline.Mapper import Mapper
from Pipeline.DataProcessor.processor import Processor
from .Exceptions.pipelineException import PipelineException
from .Learner.Models.abstractModel import AbstractModel
from .Learner.learner import Learner
from .DataProcessor.DataSplitting.splitter import Splitter
from .Learner.Models.model_loader import load_model


def load_pipeline(file: str) -> 'Pipeline':
    """
        Loads the pipeline from a file where it was previously saved
    :param file: path to the file where the pipeline was previously saved
    :return: the pipeline
    """
    mapper = Mapper("Pipeline", file)
    return Pipeline(mapper=mapper)


class Pipeline:
    """
        Represents the core of the program.
        Aims to convert raw data to trained model.
        Pipeline steps:
            1. Data cleaning & feature engineering module.
            2. #TODO complete with other modules


        Methods:
            - process: processes a dataset according to the specifications in the config file
            - convert: converts data according to the rules learnt from a previous process call
            - learn: given a dataset and a configuration it fits a model to the data
            - predict: provided that the pipeline has previously learnt a model, it predicts the output of data
            - fit: does all the steps activated in the configuration file
            - save: saves the pipeline (including the model) to a file
            - get_model: returns the model (none if it has not learnt a model previously)
            - load_pipeline(defined outside the class): reads a saved pipeline from a file and returns it
    """

    def __init__(self, config: dict = None, mapper_file: str = None, mapper: 'Mapper' = None,
                 default_config_path: str = None):
        """
            Inits the pipeline
        :param config: configuration dictionary
        :param mapper_file: the file where the mapper is saved, if existing
        :param mapper:the dictionary (in Mapper format) containing the data previously saved by the Pipeline instance
        :param default_config_path: if the pipeline is used with a configuration file located elsewhere than
                    the default location; if provided, this path will be used when creating the configuration
        Usage:
            if provided any data, the Pipeline will init itself from that dictionary
            otherwise, if provided a config it will use that, if not it will try to read the config from file
                       if a mapper file is provided the processor will be initialized with that
        """

        # data processing attributes
        if mapper is None:  # initialized by the user
            self._processor = None
            self._mapper_file = None

            if config is None:
                self._config = Pipeline._read_config_file(default_config_path)
            else:
                self._config = config
            if self._config.get("DATA_PROCESSING", False):
                self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), file=mapper_file)

            self._mapper = Mapper("Pipeline")

            # learner attributes
            self._learner = Learner(self._config.get("TRAINING_CONFIG", {}))
            self._model = None

        else:  # initialized by the load_pipeline method
            self._config = mapper.get("CONFIG", default={})
            self._processor = Processor(self._config, data=mapper.get_mapper("PROCESSOR_DATA", {}))

            model_map = mapper.get("MODEL", default=None)
            if model_map is None:
                self._model = None
            else:
                self._model = load_model(model_map)

    def process(self, data: DataFrame) -> DataFrame:
        """
            Processes the data according to the configuration in the config file
        :param data: DataFrame containing the raw data that has to be transformed
        :return: DataFrame with the modified data
        """
        start = time.time()

        result = data

        # 1. Data processing
        if self._config.get("DATA_PROCESSING", False):
            result = self._processor.process(result)

        end = time.time()
        print("Processed in {0:.4f} seconds.".format(end - start))
        return result

    def convert(self, data: DataFrame) -> DataFrame:
        """
            Converts the data to the representation previously learned by the DataProcessor
        :param data: DataFrame containing data similar to what the
        :return: DataFrame containing the converted data
        :exception: PipelineException
        """
        start = time.time()

        result = data
        if self._processor is None:
            if self._mapper_file is None:
                raise PipelineException(
                    "Mapper file not set. In order to convert data, provide a mapper file to the constructor.")
            self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), self._mapper_file)
        result = self._processor.convert(data)
        end = time.time()
        print("Converted in {0:.4f} seconds.".format(end - start))
        return result

    def learn(self, data: DataFrame, y_column: str = None) -> AbstractModel:
        """
            Learns a model from the data.
        :param y_column: the name of the predicted column
        :param data: DataFrame containing the dataset to learn
        :return: trained model or None if trained is not set to true in config
        """
        start = time.time()

        if y_column is None:
            y_column = self._config.get("TRAINING_CONFIG", {}).get("PREDICTED_COLUMN_NAME", "undefined")

        result = None
        # 2. Model learning
        if self._config.get("TRAINING", False):
            x, y = Splitter.XYsplit(data, y_column)

            result = self._learner.learn(X=x, Y=y)

        end = time.time()
        print("Learnt in {0:.4f} seconds.".format(end - start))
        self._model = result
        return result

    def predict(self, data: DataFrame) -> DataFrame:
        """
            Predicts the output of the data using a previously learnt module.
        :param data: DataFrame with the x values to be predicted
        :return: DataFrame with the predicted values
        :exception PipelineException when no model has been previously learnt
        """
        if self._model is None:
            raise PipelineException("Could not predict unless a training has been previously done.")

        return self._model.predict(data)

    def fit(self, data: DataFrame):
        """
            Completes the pipeline as specified in the configuration file.
        :param data: DataFrame with raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """

        # Iterating over the pipeline steps
        # 1. Data processing
        result = self.process(data)

        # 2. Learning
        result = self.learn(result)

        return result

    def __call__(self, data: DataFrame):
        """
            Calls the fit method by calling the pipeline.
        :param data: DataFrame with raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """
        return self.fit(data)

    def save(self, file: str) -> 'Pipeline':
        """
            Saves the pipeline logic to the specified file for further reusage.
        :return: None
        """
        # save the initial configuration for further operations on the pipeline
        self._mapper.set("CONFIG", self._config)

        # save the processor mapper for further data processing or conversion
        self._mapper.set_mapper(self._processor.get_mapper(), "PROCESSOR_DATA")

        # save the model to file
        model_map = None
        if not (self._model is None):
            model_map = self._model.to_dict()

        self._mapper.set("MODEL", model_map)

        # save the mapper to file
        self._mapper.save_to_file(file)

        return self

    def get_model(self) -> AbstractModel:
        """
            Returns the trained model or None is no training has been done
        :return:
        """
        return self._model

    @staticmethod
    def _read_config_file(path: str = None) -> dict:
        """
            Reads the default configuration file
        :param path: the explicit path for the configuration file
        :return: dictionary with the encodings
        """
        if path is None:
            path = os.path.join(os.getcwd(), 'Pipeline', 'config.json')

        # TODO add error handling for incorrect path

        # print(path)
        with open(path) as json_file:
            data = json.load(json_file)
        return data
