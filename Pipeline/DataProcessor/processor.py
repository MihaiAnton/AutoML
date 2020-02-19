from pandas import DataFrame
from pandas import concat


from .DataCleaning import Cleaner
from .DataSplitting import Splitter
from .FeatureEngineering import Engineer
from .FeatureMapping import Mapper
from ..Exceptions.dataProcessorException import DataProcessorException

class Processor:
    """
        Data processing module; the first component of the pipeline.
        Converts data from raw format to a representation that can understood by further learning algorithms.
        Unless a configuration is passed as an argument the default one is used.
    """

    def __init__(self, config=None, file=None, data=None):
        """
            Inits the data processor with the configuration parsed from the json file
            Usage: pass a mapper dictionary and the processor will init itself from that
                   otherwise, pass the configuration dictionary and, optionally, a file with the saved mapper
        :param config: configuration dictionary that contains the logic of processing data

        """
        if data is None:
            if file is None:
                self._mapper = Mapper("Processor")          #maps the changes in the raw data, for future prediction tasks
            else:
                self._mapper = Mapper("Processor", file=file)

            if config is None:
                config = self._mapper.get("PROCESSOR_CONFIG",{})
            self._config = config

        else:
            if config is None:
                config = {}
            self._mapper = Mapper("Processor", dictionary=data)
            self._config = self._mapper.get("PROCESSOR_CONFIG",config)

    def get_data(self)->dict:
        """
            Returns the mapper dictionary, for file saving purposes
        :return: dict
        """
        return self._mapper.get_map()


    def process(self, data: DataFrame):
        """
            Completes the whole cycle of automated feature engineering.
            Received raw data and returns data ready to be fed into the next step of the pipeline or into a learning algorithm.
        :param data: Raw data input, in form of DataFrame
        :return: cleaned and processed data, in form of DataFrame
        :exception: DataProcessorException
        """

        if self._config.get("NO_PROCESSING",True):      #no processing configured in the configuration file
            self._mapper.set("NO_PROCESSING",True)
            return data

        ## go over all the steps in the data processing pipeline

        # 1. Data cleaning
        if self._config.get("DATA_CLEANING", False):    #data cleaning set to be done
            self._mapper.set("DATA_CLEANING",True)
            cleaner = Cleaner(self._config.get("DATA_CLEANING_CONFIG", {}))
            y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
            data = cleaner.clean(data, self._mapper, y_column)

        # 2. Data splitting
        y_column = self._config.get('PREDICTED_COLUMN_NAME', None)
        result = Splitter.XYsplit(data, y_column)
        if result is None:
            raise DataProcessorException("Expected (X,Y) tuple of DataFrames from XYsplit but got None instead")

        X,Y = result        #init the X and Y variables

        # 3. Feature engineering
        if self._config.get("FEATURE_ENGINEERING", False):  #feature engineering set to be done
            self._mapper.set("FEATURE_ENGINEERING", True)

            engineer = Engineer(self._config.get("FEATURE_ENGINEERING_CONFIG", {}))
            X = engineer.process(X, self._mapper,{})

        # 4. Retrieve mappings
        # mappings are already in the mapper field, which would be saved to file as soon as the save_processor is called

        # 5. Create the output
        data = concat([X, Y], axis=1)

        return data

    def convert(self, data: DataFrame):
        """
            Converts data to a format previously determined by the process method.
            Used after data processing for further predictions.
        :param data: Raw data input for prediction purpose
        :return: data transformed in a format previously determined by the logic within process method
        :exception:
        """
        if self._mapper.get("NO_PROCESSING", False):
            return data

        ## go over all the steps in the data processing pipeline
        # 1. Data cleaning
        if self._mapper.get("DATA_CLEANING", False):  # data cleaning set to be done
            data = Cleaner.convert(data, self._mapper)

        # 2. Feature engineering
        if self._mapper.get("FEATURE_ENGINEERING", False):  # feature engineering set to be done
            data = Engineer.convert(data, self._mapper)

        return data


    def save_processor(self, file: str):
        """
            Saves the processor logic to disc.
        :param file: text file for saving the data
        :return: None on error | processor on success for chaining reasons
        :exception: DataProcessorException
        """
        try:
            self._mapper.set("PROCESSOR_CONFIG", self._config)
            self._mapper.save_to_file(file)
        except:
            raise DataProcessorException("Error while saving processor to file {}.".format(file))

    @staticmethod
    def load_processor(file:str):
        """

        :param file: the file where a processor has been previously saved with the save_processor method
        :return: the instance of a processor class with the logic within the file
        :exception:
        """
        #the file contains the mapper, and withing the mapper it already exists a configuration
        return Processor(file = file)



























