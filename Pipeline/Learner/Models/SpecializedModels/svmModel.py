import pickle
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from random import randint, randrange
import warnings

from ..abstractModel import AbstractModel
from ..modelTypes import SVM_MODEL
from ..constants import REGRESSION, CLASSIFICATION, AVAILABLE_TASKS
from ....Exceptions import SvmModelException


class SvmModel(AbstractModel):
    """
        AbstractModel implementation using a SVM actual model.
        The framework used is Sklearn
    """

    def __init__(self, task: str = "", config: dict = None, predicted_name: list = None,
                 dictionary=None):
        """
            Initializes a random forest model.
        :param config: the configuration dictionary: expected to receive the SVM_CONFIG dictionary
        :param predicted_name: the name of the the predicted column
        :param dictionary: the mapping of data (previously returned by to_dict
        """
        if type(dictionary) is dict:  # for internal use;
            self._init_from_dictionary(dictionary)  # load from a dictionary when loading from file the model
            return

        # model metadata
        self._task = task
        self._config = config
        if config is None:
            self._config = {}
        self._predicted_name = predicted_name

        # actual model
        self._model = None

        # print data
        self._configured = False
        self._kernel = None
        self._regularization = None

    # noinspection DuplicatedCode
    def train(self, X: DataFrame, Y: DataFrame, train_time: int = 600, callbacks: list = None,
              validation_split: float = 0.2) -> 'AbstractModel':
        """
                Trains the model with the data provided.
            :param validation_split: how much from the data(as percentage) should be used as validation
            :param callbacks: a list of predefined callbacks that get called at every epoch
            :param time: time of the training session in seconds: default 10 minutes
            :param X: the independent variables in form of Pandas DataFrame
            :param Y: the dependents(predicted) values in form of Pandas DataFrame
            :return: the model
        """

        if self._predicted_name is None:
            self._predicted_name = list(Y.columns)

        if self._task not in AVAILABLE_TASKS:
            self._task = self._determine_task_type(Y)

        if validation_split is None:
            x_train = X.to_numpy()
            y_train = Y.to_numpy()

            print("Training on {} samples...".format(len(y_train)))
        else:

            if type(validation_split) != float:
                raise SvmModelException("Parameter validation_split should be None or float in range [0,1)")
            if validation_split < 0 or validation_split >= 1:
                validation_split = 0.2
                warnings.warn("SvmModel: configured validation percentage is out of bounds; using default value 0.2",
                              RuntimeWarning)

            x_train, x_val, y_train, y_val = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=validation_split,
                                                              random_state=randrange(2048))

            print("Training on {} samples. Validating on {}...".format(len(y_train), len(y_val)))

        # no time tracking for this model type, since SVM's are trained only once
        # train the model
        if self._model is None:
            self._model = self._get_model()
            self._configured = True
            self._kernel = self._model.kernel
            self._regularization = self._model.C

        self._model.fit(x_train, y_train.ravel())

        if self._task == CLASSIFICATION:
            loss_name = "accuracy"
        else:
            loss_name = "loss"

        if not (validation_split is None):
            criterion_train = self._model.score(x_train, y_train)
            criterion_val = self._model.score(x_val, y_val)

            print("Training finished - Training {}: {} - Validation {}: {}"
                  .format(loss_name, criterion_train, loss_name, criterion_val))
        else:
            criterion_train = self._model.score(x_train, y_train)
            print("Training finished - Training {}: {}".format(loss_name, criterion_train))

    def predict(self, X: DataFrame) -> DataFrame:
        """
                Predicts the output of X based on previous learning
            :param X: DataFrame; the X values to be predicted into some Y Value
            :return: DataFrame with the predicted data
        """
        if self._model is None:
            raise SvmModelException("Could not call predict before train.")

        data = X.to_numpy()
        pred = self._model.predict(data)

        df = DataFrame(pred, columns=self._predicted_name)

        return df

    def model_type(self) -> str:
        """
                Returns th0e model type from available model types in file "model_types.py"
            :return: string with the model type
        """
        return SVM_MODEL

    def to_dict(self) -> dict:
        """
                Returns a dictionary representation that encapsulates all the details of the model
            :return: dictionary with 2 mandatory keys : MODEL_TYPE, MODEL_DATA
        """
        # !!! should match _init_from_dictionary loading format
        # get the model data
        model = pickle.dumps(self._model)

        data = {
            "MODEL": model,
            "METADATA": {
                "PREDICTED_NAME": self._predicted_name,
                "CONFIG": self._config,
                "TASK": self._task,
                "CONFIGURED": self._configured,
                "KERNEL": self._kernel,
                "REGULARIZATION": self._regularization
            }
        }

        return {
            "MODEL_TYPE": self.model_type(),
            "MODEL_DATA": data
        }

    def _init_from_dictionary(self, dictionary: dict):
        """
            Initializes a SvmModel from a dictionary previously returned by to_dict
        :param dictionary: mapping of previously saved data
        :return: None
        """
        # !!! should match to_dict loading format
        data = dictionary.get("MODEL_DATA")

        mdata = data.get("METADATA")
        model = data.get("MODEL")

        # init the data
        self._predicted_name = mdata.get("PREDICTED_NAME")
        self._config = mdata.get("CONFIG")
        self._task = mdata.get("TASK")
        self._configured = mdata.get("CONFIGURED")
        self._kernel = mdata.get("KERNEL")
        self._regularization = mdata.get("REGULARIZATION")

        # init the model
        self._model = pickle.loads(model)

    def _get_model(self):
        """
            Returns a model as specified in the configuration
        :return: the sklearn model
        """
        if self._task == CLASSIFICATION:
            return self._get_classifier()
        elif self._task == REGRESSION:
            return self._get_regressor()
        else:
            return None

    def _get_classifier(self) -> SVC:
        """
            Returns a sklearn SVC configured according to the configuration file
        :return: SVR
        """
        return SVC(
            C=self._config.get("REGULARIZATION_C", 1.0),
            kernel=self._config.get("KERNEL", "rbf"),
            degree=self._config.get("POLY_DEGREE", 3),
            gamma=self._config.get("GAMMA", 'auto'),
            tol=1e-6,
            cache_size=300,
            decision_function_shape=self._config.get("DECISION_FUNCTION_SHAPE", 'ovr'),
            random_state=randint(1, 1024)
        )

    def _get_regressor(self) -> SVR:
        """
            Returns a sklearn SVR configured according to the configuration file
        :return: SVR
        """
        return SVR(
            C=self._config.get("REGULARIZATION_C", 1.0),
            kernel=self._config.get("KERNEL", "rbf"),
            degree=self._config.get("POLY_DEGREE", 3),
            gamma=self._config.get("GAMMA", 'auto'),
            tol=1e-6,
            cache_size=300
        )

    def _description_string(self) -> str:
        if self._configured is False:
            return "SVM - Not Configured"
        else:
            task = "Classifier" if self._task == CLASSIFICATION else "Regression"
            return "SVM {task} - Kernel: {kernel} | Regularization: {reg}".format(
                task=task,
                kernel=self._kernel,
                reg=self._regularization
            )

    def get_config(self) -> dict:
        return self._config