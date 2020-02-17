from pandas import DataFrame
from ...Exceptions.dataProcessorException import DataSplittingException

class Splitter:
    """
        Handles the data split logic.
    """

    @staticmethod
    def XYsplit(data:DataFrame, y_column):
        """
            Splits the data in X and Y.
        :param data: dataframe with the dataset
        :param y_column: the name of the predicted column
        :return: tuple like (X,Y), where both are dataframes | None on error
        :exception: DataSplittingException
        """
        if y_column in data.columns:
            Y = data[[y_column]]
            X_cols = data.columns.tolist()
            X_cols.remove(y_column)         #removing the y columns from the x subset
            X = data[X_cols]
            return X,Y
        else:
            raise DataSplittingException("Cannot split after non-existing Y column {}".format(y_column))


