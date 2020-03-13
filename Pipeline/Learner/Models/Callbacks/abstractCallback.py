from ..abstractModel import AbstractModel
from abc import abstractmethod, ABC


class AbstractCallback(ABC):
    """
        Defines what a callback should behave like.
        A callback has a model and a frequency of calling.
    """

    def __init__(self, model: AbstractModel = None, frequency: int = 1):
        """
            Inititalizes an AbstractCallback object
        :param model: the model on which the callback is executed. If None, the object that the callback
            is passed on will be used
        :param frequency: number of epochs to be called at (by default at every epoch)
        """
        self._model = model
        self._frequency = frequency

    def set_model(self, model: AbstractModel):
        """
            Sets the model if not already explicitly set another one
        :param model: AbstractModel implementation
        :return: None
        """
        if self._model is None:
            self._model = model

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
            Calls the callback, with the logic to be implemented within the actual implementation
        :param args: any number of objects as defined in the actual implementation
        :param kwargs: any number of mapped objects as defined in the actual implementation
        :return: None
        """
