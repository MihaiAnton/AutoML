from ..abstractCallback import AbstractCallback


class EarlyStopping(AbstractCallback):

    def __init__(self):
        AbstractCallback.__init__(self)

    def __call__(self, *args, **kwargs):
        """
            This gets called every time an epoch finishes training.
        :param args:
        :param kwargs:
        :return:
        """
        # TODO
