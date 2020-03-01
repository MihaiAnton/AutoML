from pandas import DataFrame

class Model:
    """
        The product of the learning phase.
        It's main task is to predict after a training session has been done
    """

    def __init__(self):
        """
        
        """


    def predict(self, X:DataFrame)->DataFrame:
        """
            Predicts the output of X based on previous learning
        :param X: DataFrame; the X values to be predicted into some Y Value
        :return: DataFrame with the predicted data
        """

        return X