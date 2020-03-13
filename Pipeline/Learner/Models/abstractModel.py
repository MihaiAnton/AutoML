import pickle
from abc import ABC, abstractmethod

from pandas import DataFrame


class AbstractModel(ABC):
    """
        The end result of the pipeline.
        It's main task is to predict after a training session has been done.

        Methods:
            - train: trains the actual model based on a dataset
            - predict: predicts the output of a dataset
            - to_dict: returns a serializable dictionary
            - save: saves the model to file
            - model_type: returns the model type, as defined in "SpecializedModel/modelTypes.py"

        Behaviour:
            - calling an object ( model_instance(data) ), will return the prediction
    """

    @abstractmethod
    def train(self, X: DataFrame, Y: DataFrame, time: int = 600, callbacks: list = None) -> 'AbstractModel':
        """
            Trains the model with the data provided.
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :return: the model
        """

    @abstractmethod
    def predict(self, X: DataFrame) -> DataFrame:
        """
            Predicts the output of X based on previous learning
        :param X: DataFrame; the X values to be predicted into some Y Value
        :return: DataFrame with the predicted data
        """

    def __call__(self, X: DataFrame) -> DataFrame:
        """
            Calls the predict method.
        :param X: data to be predicted
        :return: predicted data
        """
        return self.predict(X)

    @abstractmethod
    def to_dict(self) -> dict:
        """
            Returns a dictionary representation that encapsulates all the details of the model
        :return: dictionary with 2 mandatory keys : MODEL_TYPE, MODEL_DATA
        """

    def save(self, file: str):
        """
            Saves the model to file
        :param file: the name of the file or the absolute path to it
        :return: self for chaining purposes
        """
        import json
        with open(file, 'wb') as f:
            data = self.to_dict()
            pickle.dump(data, f)
        return self

    @abstractmethod
    def model_type(self) -> str:
        """
            Returns the model type from available model types in file "model_types.py"
        :return: string with the model type
        """

    @staticmethod
    def _determine_task_type(Y: DataFrame) -> str:
        """
            Determines heuristically the task type given the output variable.
        :return: string from constants.py/AVAILABLE_TASKS with the specific task
        """

        total_number = len(Y)
        unique_number = len(Y.drop_duplicates(ignore_index=True))

        if unique_number / total_number > 0.08:  # there have to be at least 8% unique values from the total number
            return "regression"  # of values in order to be considered regression
        else:
            return "classification"

    @staticmethod
    def _categorical_mapping(data: DataFrame) -> dict:
        """
            Checks all the unique columns, creates categorical features and returns mapping
            Return type dict {
                new_col_name : {
                    column1: value1,
                    column2: value2
                    ...
                }
                ...
            }
            The returned type represents how an entry should be in order to be part of one column
        :param data: the output variable to be encoded
        :return: the mapping dictionary
        """
        class_mappings = {}

        uniques = data.drop_duplicates(ignore_index=True)  # get the unique rows

        for row in uniques.iterrows():
            row = row[1]
            values = {}
            for col in data.columns.to_list():  # for each unique row get the values that determine it
                values[col] = row[col]

            new_class_name = '&'.join([key + "_" + str(values[key]) for key in values.keys()])  # get a new class
            # name reuniting all
            class_mappings[new_class_name] = values  # set the values to the new created class name

        return class_mappings

    @staticmethod
    def _to_categorical(data: DataFrame, mapping: dict) -> DataFrame:
        """
            According to the dictionary previously created, returns a converted dataset
        :param mapping: the mapping dictionary created with method _categorical_mapping on similar dataset
        :param data: dataset to be converted
        :return: converted dataset
        """
        new_columns = list(mapping.keys())
        new_values = []

        for row in data.iterrows():  # for each entry in the dataset

            final_column = None
            for possible_col in sorted(new_columns):  # check for every possible column
                ok = True
                for column in list(mapping[possible_col].keys()):  # for evey column, check if it matches the condition
                    if mapping[possible_col][column] != row[1][column]:
                        ok = False
                        break

                if ok:  # if it matches, set the column
                    final_column = possible_col
                    break

            d = {col: 0 for col in new_columns}  # set the row
            if not (final_column is None):
                d[final_column] = 1
            new_values.append(d)

        return DataFrame(new_values)

    @staticmethod
    def _from_categorical(data: DataFrame, mapping: dict) -> DataFrame:
        """
            Based on the mapping computed with _categorial_mapping function on a similar dataset,
        converts the encoded data into initial data
        :param data: dataset to be converted back to the inital form
        :param mapping: the mapping computed with _categorical mapping function
        :return: reverted dataset
        """
        categories = data.idxmax(axis=1)  # get the categories
        return DataFrame([mapping[c] for c in categories])  # easily construct the dataframe from list of
        # dictionaries
