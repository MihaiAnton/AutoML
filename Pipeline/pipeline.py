import json
import os

from pandas import DataFrame

from Pipeline.DataProcessor.processor import Processor


class Pipeline:
    """
        Represents the core of the program.
        Aims to convert raw data to trained model.
        Pipeline steps:
            1. Data cleaning & feature engineering module.
            2. #TODO complete with other modules
    """

    def __init__(self, config=None):
        """
            Inits the pipeline
        :param config: configuration dictionary
        """
        if config is None:
            self._config = Pipeline._read_config_file()
        else:
            self._config = config
        if self._config.get("DATA_PROCESSING", False):
            self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"))

    def fit(self, data: DataFrame):
        """
            Completes the pipeline as specified in the configuration file.
        :param data: raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """

        result = data
        #Iterating over the pipeline steps
        # 1. Data processing
        if self._config.get("DATA_PROCESSING", False):
            result = self._processor.process(result)

        # must be deleted later
        result.to_csv("Datasets/titanic_generated.csv")


        # 2. #TODO

        return result

    @staticmethod
    def _read_config_file():
        with open(os.getcwd()+'\Pipeline\config.json') as json_file:
            data = json.load(json_file)
        return data































