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


class Pipeline:
    """
        Represents the core of the program.
        Aims to convert raw data to trained model.
        Pipeline steps:
            1. Data cleaning & feature engineering module.
            2. #TODO complete with other modules
    """

    def __init__(self, config: dict = None, mapper_file: str = None, mapper: 'Mapper' = None):
        """
            Inits the pipeline
        :param config: configuration dictionary
        :param mapper_file: the file where the mapper is saved, if existing
        :param mapper:the dictionary (in Mapper format) containing the data previously saved by the Pipeline instance
        Usage:
            if provided any data, the Pipeline will init itself from that dictionary
            otherwise, if provided a config it will use that, if not it will try to read the config from file
                       if a mapper file is provided the processor will be initialized with that
        """
        if mapper is None:  # initialized by the user
            self._processor = None
            self._mapper_file = None

            if config is None:
                self._config = Pipeline._read_config_file()
            else:
                self._config = config
            if self._config.get("DATA_PROCESSING", False):
                self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), file=mapper_file)

            self._mapper = Mapper("Pipeline")

        else:  # initialized by the load_pipeline method
            self._config = mapper.get("CONFIG", default={})
            self._processor = Processor(self._config, data=mapper.get_mapper("PROCESSOR_DATA", {}))

        self._learner = Learner(self._config.get("TRAINING_CONFIG", {}))

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
        print("Processed in {} seconds.".format(end - start))
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
            X, Y = Splitter.XYsplit(data, y_column)

            result = self._learner.learn(X=X, Y=Y)

        end = time.time()
        print("Learnt in {} seconds.".format(end - start))
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
        print("Converted in {} seconds.".format(end - start))
        return result

    def fit(self, data: DataFrame):
        """
            Completes the pipeline as specified in the configuration file.
        :param data: raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """

        # Iterating over the pipeline steps
        # 1. Data processing
        result = self.process(data)

        # 2. Learning
        result = self.learn(result)

        return result

    def save(self, file: str) -> 'Pipeline':
        """
            Saves the pipeline logic to the specified file for further reusage.
        :return: None
        """
        self._mapper.set_mapper(self._processor.get_mapper(), "PROCESSOR_DATA")
        self._mapper.set("CONFIG", self._config)
        self._mapper.save_to_file(file)
        # data = {                                    #the data format here should match the constructor
        #     "CONFIG":self._config,                      # because it will try to reconstruct the Pipeline
        #     "PROCESSOR_DATA":processor_data             # provided this configuration
        # }
        #
        # with open(file, 'w') as f:
        #     json.dump(data, f)
        # return self

    @staticmethod
    def load_pipeline(file: str) -> 'Pipeline':
        """
            Loads the pipeline from a file where it was previously saved
        :param file: path to the file where the pipeline was previously saved
        :return: the pipeline
        """
        mapper = Mapper("Pipeline", file)
        return Pipeline(mapper=mapper)

    @staticmethod
    def _read_config_file() -> dict:
        path = os.path.join(os.getcwd(), 'Pipeline', 'config.json')
        # print(path)
        with open(path) as json_file:
            data = json.load(json_file)
        return data
