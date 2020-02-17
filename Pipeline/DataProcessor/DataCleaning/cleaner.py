from pandas import DataFrame
from ..FeatureMapping import Mapper

class Cleaner:

    def __init__(self, config={}):
        self._config = config
        self._mapper = Mapper("Cleaner")

    def clean(self, data: DataFrame, mapper: 'Mapper', predicted_col=None):
        """
            Cleans the data by removing rows/columns where necessary.
        :param predicted_col: which is the column name that we want to predict
        :param data: the raw data that needs to be cleaned
        :param mapper: the mapper class that saves all the changes
        :return: the cleaned data
        """

        #remove cols with predicted value missing
        if self._config.get('REMOVE_WHERE_Y_MISSING', False) and not(predicted_col is None):     #if it exists and if it is set on true
            if predicted_col in data.columns:
                data = data.dropna(subset=[predicted_col])

        #remove cols which are explicitly set to be removed
        cols_to_remove = self._config.get('COLUMNS_TO_REMOVE', [])
        for column in cols_to_remove:
            if column in data.columns:
                data.drop(column, axis=1, inplace=True)


        #remove lines with more than ROW_REMOVAL_THRESHOLD % missing values
        if self._config.get('REMOVE_ROWS', False):
            column_count = len(data.columns)
            remove_threshold = float(self._config.get('ROW_REMOVAL_THRESHOLD', 1))
            data = data[data.isna().sum(axis=1) <= column_count-remove_threshold*column_count]  #filter out the ones that have too many missing values

        if self._config.get('REMOVE_COLUMNS', False):
            row_count = len(data.index)
            remove_threshold = float(self._config.get('COLUMN_REMOVAL_THRESHOLD', 1))
            cols_to_drop = data.columns[data.isna().sum()>=row_count*remove_threshold].tolist()

            #mark the deleted columns
            self._mapper.set("RemovedColumns", cols_to_drop)

            data = data.drop(cols_to_drop, axis=1)

        #set the mapper
        mapper.set_mapper(self._mapper)

        data.reset_index(drop=True, inplace=True)
        return data






    def convert(self, data:DataFrame, mapper: 'Mapper'):
        """
            Based on the mapping determined in the clean method, it cleans the input data accordingly.
        :param mapper: mapper class instance that holds all the changes that have to be done to the dataset
        :param data: the raw input that needs to be converted
        :return: the converted data
        """

































