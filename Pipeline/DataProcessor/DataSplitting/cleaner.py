from pandas import DataFrame
from Pipeline.DataProcessor.FeatureMapping.mapper import Mapper

class Cleaner:

    def __init__(self, config=None):
        pass

    def clean(self, data: DataFrame, mapper: 'Mapper'):
        """
            Cleans the data by removing rows/columns where necessary.
        :param data: the raw data that needs to be cleaned
        :param mapper: the mapper class that saves all the changes
        :return: the cleaned data
        """


    def convert(self, data:DataFrame, mapper: 'Mapper'):
        """
            Based on the mapping determined in the clean method, it cleans the input data accordingly.
        :param mapper: mapper class instance that holds all the changes that have to be done to the dataset
        :param data: the raw input that needs to be converted
        :return: the converted data
        """





