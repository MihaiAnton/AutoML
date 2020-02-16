from pandas import DataFrame


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
        :exception: TODO add exception
        """
        if y_column in data.columns:
            Y = data[[y_column]]
            X_cols = data.columns.tolist()
            X_cols.remove(y_column)         #removeing the y columns from the x subset
            X = data[X_cols]
            return X,Y
        else:
            #TODO raise exception
            return None


