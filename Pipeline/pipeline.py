import json
import os
import time
from pandas import DataFrame

from Pipeline.DataProcessor.processor import Processor
from .Exceptions.pipelineException import PipelineException


class Pipeline:
    """
        Represents the core of the program.
        Aims to convert raw data to trained model.
        Pipeline steps:
            1. Data cleaning & feature engineering module.
            2. #TODO complete with other modules
    """

    def __init__(self, config=None, mapper_file=None, data=None):
        """
            Inits the pipeline
        :param config: configuration dictionary
        """
        if data is None:                #initialized by the user
            self._processor = None
            self._mapper_file = None

            if config is None:
                self._config = Pipeline._read_config_file()
            else:
                self._config = config
            if self._config.get("DATA_PROCESSING", False):
                self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), file=mapper_file)

        else:                           #initialized by the load_pipeline method
            self._config = data["CONFIG"]
            self._processor = Processor(self._config,data=data["PROCESSOR_DATA"])



    def process(self, data: DataFrame):
        """
            Processes the data according to the configuration in the config file
        :param data: DataFrame containing the raw data that has to be transformed
        :return: DataFrame with the omdified data
        """
        start = time.time()

        result = data
        # Iterating over the pipeline steps
        # 1. Data processing
        if self._config.get("DATA_PROCESSING", False):
            result = self._processor.process(result)

        end = time.time()
        print("Processed in {} seconds.".format(end-start))
        return result

    def convert(self, data: DataFrame):
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

        result = data
        # Iterating over the pipeline steps
        # 1. Data processing
        result = self.process(data)

        # must be deleted later
        # result.to_csv("Datasets/titanic_generated.csv", index=False)

        # 2. #TODO

        return result

    def save(self, file):
        """
            Saves the pipeline logic to the specified file for further reusage.
        :return: None
        """
        processor_data = self._processor.get_data()


        data = {
            "CONFIG":self._config,
            "PROCESSOR_DATA":processor_data
        }
        with open(file, 'w') as f:
            json.dump(data, f)
        return self

    @staticmethod
    def load_pipeline(file: str) -> 'Pipeline':
        """
            Loads the pipeline from a file where it was previously saved
        :param file: path to the file where the pipeline was previously saved
        :return: the pipeline
        """
        with open(file) as f:
            data = json.load(f)
            return Pipeline(data=data)

    @staticmethod
    def _read_config_file():
        with open(os.getcwd() + '\Pipeline\config.json') as json_file:
            data = json.load(json_file)
        return data
