from confounds.base import BaseDeconfound


class Harmonize(BaseDeconfound):
    """
    Estimator to transform the input features to harmonize the input features
    across a given set of confound variables.

    Example methods include:
    Scaling (global etc)
    Normalization (Quantile, Functional etc)
    Surrogate variable analysis
    ComBat

    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='Harmonize')

        raise NotImplementedError()
