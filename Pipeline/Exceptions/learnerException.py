
class ModelSelectionException(Exception):
    """
        Exception for the model selection
    """

    def __init__(self, message):
        super().__init__(message)

    def __repr__(self):
        return "ModelSelectionException: {}.".format(self._message)

class DeepLearningModelException(Exception):
    """
        Generic exception for the neural network
    """

    def __init__(self, message):
        super().__init__(message)

    def __repr__(self):
        return "NeuralNetworkModelException: {}.".format(self._message)

