from confounds.base import BaseDeconfound


class StratifyByConfounds(BaseDeconfound):
    """
    Sub- or re-sampling procedures to minimize the confound-to-target correlation.
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='StratifyByConfounds')

        raise NotImplementedError()
