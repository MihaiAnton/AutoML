import pickle
from abc import ABC, abstractmethod
from pandas import DataFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    mean_absolute_error, mean_squared_error, mean_squared_log_error

from .constants import CLASSIFICATION, REGRESSION
from ...Exceptions import AbstractModelException


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
    ACCEPTED_CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy"]
    ACCEPTED_REGRESSION_METRICS = ["mean_absolute_error", "mean_squared_error", "mean_squared_log_error"]

    METRICS_TO_FUNCTION_MAP = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,

        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mean_squared_log_error": mean_squared_log_error
    }

    @abstractmethod
    def train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, validation_split: float = 0.2,
              callbacks: list = None, verbose: bool = True) -> 'AbstractModel':
        """
            Trains the model with the data provided.
        :param validation_split: percentage of the data to be used in validation; None if validation should not be used
        :param callbacks: a list of predefined callbacks that get called at every epoch
        :param train_time: time of the training session in seconds: default 10 minutes
        :param X: the independent variables in form of Pandas DataFrame
        :param Y: the dependents(predicted) values in form of Pandas DataFrame
        :param verbose: decides whether or not the model prints intermediary outputs
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

    def eval(self, X: DataFrame, Y: DataFrame, task: str, metric: str):
        """
            Evaluates the model's performance and returns a score
        :param task: the task of the model (REGRESSION / CLASSIFICATION)
        :param X: the input dataset
        :param Y: the dataset to compare the prediction to
        :param metric: the metric used
        :return: the score
        """
        if task == REGRESSION:
            if metric in self.ACCEPTED_REGRESSION_METRICS:
                scorer = self.METRICS_TO_FUNCTION_MAP[metric]

                pred = self.predict(X)

                y_true = Y.to_numpy()
                y_pred = pred.to_numpy()
                score = scorer(y_true, y_pred)
                return score
            else:
                raise AbstractModelException("Metric {} not defined for {}.".format(metric, task))
        elif task == CLASSIFICATION:
            if metric in self.ACCEPTED_CLASSIFICATION_METRICS:
                scorer = self.METRICS_TO_FUNCTION_MAP[metric]

                pred = self.predict(X)

                y_true = Y.to_numpy()
                y_pred = pred.to_numpy()
                score = scorer(y_true, y_pred)

                return 1 - score
                # there a higher score is better, but the goal is minimization, this is why it is used 1/score
            else:
                raise AbstractModelException("Metric {} not defined for {}.".format(metric, task))
        else:
            raise AbstractModelException("Task type {} not understood.".format(task))

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
        string_dtypes = ["object", "string"]
        data_types = Y.dtypes
        for column in Y.columns:
            dtype = data_types[column]
            if str(dtype) in string_dtypes:
                return CLASSIFICATION

        total_number = len(Y)
        unique_number = len(Y.drop_duplicates(ignore_index=True))

        if unique_number / total_number > 0.08:  # there have to be at least 8% unique values from the total number
            return REGRESSION  # of values in order to be considered regression
        else:
            return CLASSIFICATION

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

    @abstractmethod
    def _description_string(self) -> str:
        """
            Returns the description string for printing on user output.
        :return: string
        """

    def __repr__(self):
        return self._description_string()

    @abstractmethod
    def get_config(self) -> dict:
        """
            Returns the configuration that was used to build the model
        :return: dictionary with the configuration
        """
