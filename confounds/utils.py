"""
Utilities and helpers to enable a systematic study of various aspects related to
confounds

"""

def confound_restricted_permutations(target, confounds):
    """
    Variant of permutation testing to restrict relabelling of the target within
    certain values/ranges of the confounds, to preserve feature-to-confound
    relationships, but break the feature-to-target relationships. Such
    permutations would help us test whether models are learning the
    confound-related or target-related info i.e. if the prediction performance is
    high even when target is permuted, we can infer model is learning the
    confound-related info.

    Parameters
    ----------
    target
    confounds

    Returns
    -------

    """

    raise NotImplementedError()


def score_stratified_by_confound(score, confounds):
    """
    Helper to summarize the performance score (accuracy, MSE, MAE etc) for each
    level or variant of confound. This is helpful to assess any bias towards a
    particular value when confounds are categorical (such as site or gender). So
    if the MSE (of target) for Females is much lower compared to Males, then it
    may indicate a potential bias of the model towards Females (due to imbalance in
    size?)

    Parameters
    ----------
    score
    confounds

    Returns
    -------

    """

    raise NotImplementedError()


def get_model(name='linear'):
    """String to Estimator"""

    name = name.lower()
    if name in ('linear', 'linearregression'):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif name in ('ridge', 'kernelridge'):
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge()
    elif name in ('gpr', 'gaussianprocessregression'):
        from sklearn.gaussian_process import GaussianProcessRegressor
        model = GaussianProcessRegressor()
    else:
        raise ValueError('Unrecognized model name! '
                         'Choose one of linear, ridge and GPR.')

    return model
