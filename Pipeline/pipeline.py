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
from .configuration_manager import complete_configuration


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

    # define possible states - the state in which the pipeline is at a given moment
    DYNAMIC_CALL = False  # calling the object behaves differently depending on the state of the pipeline if true
    # if set to false, the flow will respect the configuration
    STATE_MACRO = "PIPELINE_STATUS"
    RAW_STATE = "RAW_STATE"  # the pipeline is new and no operation has been done to it
    PROCESSED_STATE = "PROCESSED_STATE"  # the pipeline has been used to process data
    LEARNT_STATE = "LEARNT_STATE"
    CONVERTED_STATE = "CONVERTED_STATE"  # the pipeline has been last used to convert data
    PREDICTED_STATE = "PREDICTED_STATE"  # the pipeline has last been used to predict data

    def __init__(self, config: dict = None, mapper_file: str = None, mapper: 'Mapper' = None,
                 default_config_path: str = None, **kwargs):
        """
            Inits the pipeline
        :param config: configuration dictionary
        :param mapper_file: the file where the mapper is saved, if existing
        :param mapper:the dictionary (in Mapper format) containing the data previously saved by the Pipeline instance
        :param default_config_path: if the pipeline is used with a configuration file located elsewhere than
                    the default location; if provided, this path will be used when creating the configuration
        :param kwargs
                - include "dynamic_call=True" in the argument list to enable dynamic pipeline call
        Usage:
            if provided any data, the Pipeline will init itself from that dictionary
            otherwise, if provided a config it will use that, if not it will try to read the config from file
                       if a mapper file is provided the processor will be initialized with that
        """
        if kwargs.get("dynamic_call", False):
            self.DYNAMIC_CALL = True

        # data processing attributes
        if mapper is None:  # initialized by the user
            self._processor = None
            self._mapper_file = None

            if config is None:
                self._config = Pipeline._read_config_file(default_config_path)
            else:
                self._config = complete_configuration(config, self._read_config_file())
            if self._config.get("DATA_PROCESSING", False):
                self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), file=mapper_file)

            self._mapper = Mapper("Pipeline")

            # learner attributes
            self._learner = Learner(self._config.get("TRAINING_CONFIG", {}))
            self._model = None

        else:  # initialized by the load_pipeline method
            self._mapper = mapper
            self._config = mapper.get("CONFIG", default={})
            self._processor = Processor(self._config, data=mapper.get_mapper("PROCESSOR_DATA", {}))

            model_map = mapper.get("MODEL", default=None)
            if model_map is None:
                self._model = None
            else:
                self._model = load_model(model_map)

            self._learner = Learner(self._config.get("TRAINING_CONFIG", {}), model=self._model)

        # set status of pipeline
        current_status = self._mapper.get(self.STATE_MACRO, False)
        if current_status is False:
            self._mapper.set(self.STATE_MACRO, self.RAW_STATE)

        self._mapper.set("CONVERSION_DONE", False)

    def _record_data_information(self, data: DataFrame, source: str, *args, **kwargs) -> None:
        """
            Maps metadata about the data fed into the pipeline.
        :param data: DataFrame containing the dataset
        :param source: string containing the source of the data
                        - data is recorded on process() and on learn(), to be used later in convert() and predict()
                        - source should be either "process" or "learn"
        :return: None
        """
        info = self._mapper.get("DATA_METADATA", default={})

        if source == "process":
            new_info = {
                "shape": data.shape,
                "columns": data.columns.to_list()
            }
        elif source == "learn":
            new_info = {
                "shape": data.shape,
                "y_column": kwargs.get("y_column", "undefined")
            }
        else:
            new_info = {}

        info[source] = new_info
        self._mapper.set("DATA_METADATA", info)

    def process(self, data: DataFrame, verbose: bool = True) -> DataFrame:
        """
            Processes the data according to the configuration in the config file
        :param verbose: decides if the process() method will produce any output
        :param data: DataFrame containing the raw data that has to be transformed
        :return: DataFrame with the modified data
        """
        start = time.time()
        self._record_data_information(data, "process")
        result = data

        # 1. Data processing

        if self._config.get("DATA_PROCESSING", False):
            result = self._processor.process(result, verbose=verbose)

        self._mapper.set("X_COLUMNS_PROCESS", list(data.columns))
        self._mapper.set("CONVERSION_DONE", True)
        end = time.time()
        print("Processed in {0:.4f} seconds.".format(end - start)) if verbose else None
        self._mapper.set(self.STATE_MACRO, self.PROCESSED_STATE)
        return result

    def convert(self, data: DataFrame, verbose: bool = True) -> DataFrame:
        """
            Converts the data to the representation previously learned by the DataProcessor
        :param verbose: decides if the convert() method will produce any output
        :param data: DataFrame containing data similar to what the
        :return: DataFrame containing the converted data
        :exception: PipelineException
        """
        start = time.time()

        if self._processor is None:
            if self._mapper_file is None:
                raise PipelineException(
                    "Mapper file not set. In order to convert data, provide a mapper file to the constructor.")
            self._processor = Processor(self._config.get("DATA_PROCESSING_CONFIG"), self._mapper_file)
        result = self._processor.convert(data, verbose=verbose)
        end = time.time()
        print("Converted in {0:.4f} seconds.".format(end - start)) if verbose else None
        self._mapper.set(self.STATE_MACRO, self.CONVERTED_STATE)
        return result

    def learn(self, data: DataFrame, y_column: str = None, verbose: bool = True) -> AbstractModel:
        """
            Learns a model from the data.
        :param verbose: decides if the the learn() method should produce any output
        :param y_column: the name of the predicted column
        :param data: DataFrame containing the dataset to learn
        :return: trained model or None if trained is not set to true in config
        """
        start = time.time()
        self._record_data_information(data, "learn")

        if y_column is None:
            y_column = self._config.get("TRAINING_CONFIG", {}).get("PREDICTED_COLUMN_NAME", "undefined")

        result = None
        # 2. Model learning
        if self._config.get("TRAINING", False):
            x, y = Splitter.XYsplit(data, y_column)
            self._mapper.set("X_COLUMNS_TRAIN", list(data.columns))

            result = self._learner.learn(X=x, Y=y, verbose=verbose)

        end = time.time()
        print("Learnt in {0:.4f} seconds.".format(end - start)) if verbose else None
        self._model = result
        self._mapper.set(self.STATE_MACRO, self.LEARNT_STATE)
        return result

    def predict(self, data: DataFrame, verbose: bool = False) -> DataFrame:
        """
            Predicts the output of the data using a previously learnt module.
        :param verbose: decide is the predict method should output information to the console
        :param data: DataFrame with the x values to be predicted
        :return: DataFrame with the predicted values
        :exception PipelineException when no model has been previously learnt
        """
        if self._model is None:
            raise PipelineException("Could not predict unless a training has been previously done.")

        columns = list(data.columns)
        columns.sort()

        learnt_columns = self._mapper.get("X_COLUMNS_TRAIN", [])
        learnt_columns.sort()

        if columns == learnt_columns:  # the columns to predicts are the learnt columns
            self._mapper.set(self.STATE_MACRO, self.PREDICTED_STATE)
            return self._model.predict(data)

        elif self._mapper.get("CONVERSION_DONE", False):  # the columns differ (maybe conversion has to be done)
            processed_cols = self._mapper.get("X_COLUMNS_PROCESS", [])
            processed_cols.sort()

            if columns == processed_cols:
                converted = self.convert(data)
                return self.predict(converted)

            else:
                raise PipelineException("Expected conversion from columns {}; received {}"
                                        .format(self._mapper.get("X_COLUMNS_PROCESS", []), list(data.columns)))

        else:
            raise PipelineException("Expected model with columns {}; received {}"
                                    .format(self._mapper.get("X_COLUMNS_TRAIN", []), list(data.columns)))

    def fit(self, data: DataFrame, verbose: bool = True):
        """
            Completes the pipeline as specified in the configuration file.
        :param verbose: decides if the method fit() and all the methods called in it should produce any output
        :param data: DataFrame with raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """

        # Iterating over the pipeline steps
        # 1. Data processing
        result = self.process(data, verbose=verbose)

        # 2. Learning
        result = self.learn(result, verbose=verbose)

        return result

    def __call__(self, data: DataFrame, verbose: bool = True):
        """
            Calls the fit method by calling the pipeline.
        :param verbose: decides if the call on a pipeliene should produce any output
        :param data: DataFrame with raw data
        :return: data/ cleaned data/ processed data/ trained model ( based on the choices in the config file)
        """
        if self.DYNAMIC_CALL is False:
            return self.fit(data, verbose=verbose)

        else:  # decide what to do depending on the state
            metadata = self._mapper.get("DATA_METADATA", {})
            state = self._mapper.get(self.STATE_MACRO, self.RAW_STATE)

            if state == self.RAW_STATE:  # follow the configuration
                print("Pipeline Dynamic Call: fit()")
                return self.fit(data, verbose=verbose)

            if data.shape == metadata.get("process", {}).get("shape", ()):  # probably a conversion is wanted
                if state == self.LEARNT_STATE:
                    print("Pipeline Dynamic Call: learn()")
                    return self.learn(data, metadata.get("learn", {}).get("y_column", "undefined"))

                else:
                    print("Pipeline Dynamic Call: convert()")
                    return self.convert(data, verbose=verbose)

            elif data.shape[1] == metadata.get("process", {}).get("shape", (-1, -1))[1] - 1:
                # if a model is present -> prediction ; else -> conversion
                if self._model is None:
                    print("Pipeline Dynamic Call: convert()")
                    return self.convert(data, verbose=verbose)

                else:
                    print("Pipeline Dynamic Call: fit()")
                    return self.predict(data, verbose=verbose)

            return self.fit(data, verbose=verbose)

    def save(self, file: str, include_model: bool = True) -> 'Pipeline':
        """
            Saves the pipeline logic to the specified file for further re-usage.
        :param file: the path(or name) of the save file
        :param include_model: default true | decides whether or not the model is included in the save file with the pipeline
        :return: None
        """
        # save the initial configuration for further operations on the pipeline
        self._mapper.set("CONFIG", self._config)

        # save the processor mapper for further data processing or conversion
        self._mapper.set_mapper(self._processor.get_mapper(), "PROCESSOR_DATA")

        # save the model to file
        model_map = None
        if include_model and (not (self._model is None)):
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
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'config.json')

        if not os.path.exists(path):
            raise PipelineException("Configuration Json file could not be parsed from source {}.".format(path))

        # print(path)
        with open(path) as json_file:
            data = json.load(json_file)
        return data
