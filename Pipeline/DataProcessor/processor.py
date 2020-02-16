from pandas import DataFrame
from pandas import concat

from Pipeline.DataProcessor.DataCleaning.cleaner import Cleaner
from Pipeline.DataProcessor.DataSplitting.splitter import Splitter
from Pipeline.DataProcessor.FeatureEngineering.engineer import Engineer
from Pipeline.DataProcessor.FeatureMapping.mapper import Mapper


class Processor:
    """
        Data processing module; the first component of the pipeline.
        Converts data from raw format to a representation that can understood by further learning algorithms.
        Unless a configuration is passed as an argument the default one is used.
    """

    def __init__(self, config:dict):
        """
            Inits the data processor with the configuration parsed from the json file
        :param config: configuration dictionary that contains the logic of processing data
        """
        self._config = config
        self._mapper = Mapper("Processor")          #maps the changes in the raw data, for future prediction tasks


    def process(self, data: DataFrame):
        """
            Completes the whole cycle of automated feature engineering.
            Received raw data and returns data ready to be fed into the next step of the pipeline or into a learning algorithm.
        :param data: Raw data input, in form of DataFrame
        :return: cleaned and processed data, in form of DataFrame
        :exception: TODO
        """

        if self._config.get("NO_PROCESSING",True):      #no processing configured in the configuration file
            return data

        ## go over all the steps in the data processing pipeline

        # 1. Data cleaning
        if self._config.get("DATA_CLEANING", False):    #data cleaning set to be done
            cleaner = Cleaner(self._config.get("DATA_CLEANING_CONFIG", {}))
            y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
            data = cleaner.clean(data, self._mapper, y_column)

        # 2. Data splitting
        y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
        result = Splitter.XYsplit(data, y_column)
        if result is None:
            #TODO throw exception
            pass

        X,Y = result        #init the X and Y variables

        # 3. Feature engineering
        if self._config.get("FEATURE_ENGINEERING", False):  #feature engineering set to be done
            engineer = Engineer(self._config.get("FEATURE_ENGINEERING_CONFIG", {}))
            X = engineer.process(X, self._mapper,{})

        # 4. Retrieve mappings
        #TODO save the mappings to file

        # 5. Create the output
        data = concat([X, Y], axis=1)

        return data

    def convert(self, data: DataFrame):
        """
            Converts data to a format previously determined by the process method.
            Used after data processing for further predictions.
        :param data: Raw data input for prediction purpose
        :return: data transformed in a format previously determined by the logic within process method
        :exception: TODO
        """
        pass

    @staticmethod
    def save_processor(processor: 'Processor', file: str):
        """
            Saves the processor logic to disc.
        :param processor: previously trained processor
        :param file: text file for saving the data
        :return: None on error | processor on success for chaining reasons
        :exception: TODO
        """
        pass

    @staticmethod
    def load_processor(file):
        """
            Loads a processor from a processor file and returns the object.
        :param file: the file where a processor has been previously saved with the save_processor method
        :return: the instance of a processor class with the logic within the file
        :exception: TODO
        """
        pass



























