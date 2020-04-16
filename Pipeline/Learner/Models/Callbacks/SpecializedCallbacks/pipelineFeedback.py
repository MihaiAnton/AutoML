from ..abstractCallback import AbstractCallback


class PipelineFeedback(AbstractCallback):
    """
        Defines a callback that receives some dictionary with data from the pipeline
    and calls the abstract function with it.
    """

    def f(self, data: dict):
        if self._fun is not None:
            try:
                self._fun(data)
            except Exception:
                pass

        return data
