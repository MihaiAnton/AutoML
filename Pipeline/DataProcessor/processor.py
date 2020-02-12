from pandas import DataFrame

class Processor:
    """
        Data processing module; the first component of the pipeline.
        Converts data from raw format to a representation that can understood by further learning algorithms.
        Unless a configuration is passed as an argument the default one is used.
    """

    def __init__(self, config=None):
        pass

    def process(self, data: DataFrame):
        """
            Completes the whole cycle of automated feature engineering.
            Received raw data and returns data ready to be fed into the next step of the pipeline or into a learning algorithm.
        :param data: Raw data input, in form of DataFrame
        :return: cleaned and processed data, in form of DataFrame
        :exception: TODO
        """

        pass

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



























