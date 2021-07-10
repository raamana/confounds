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


def get_deconfounder(name='residualize'):
    """String to Deconfounding Estimator"""

    name = name.lower()
    if name in ('residualize', 'regressout',
                'residualize_linear', 'regressout_linear'):
        from confounds.base import Residualize
        est = Residualize()
    # elif name in ('residualize_ridge', 'residualize_kernelridge'):
    #     from confounds.base import Residualize
    #     est =  Residualize(model='KernelRidge')
    # elif name in ('residualize_gpr', 'residualize_gaussianprocessregression'):
    #     from confounds.base import Residualize
    #     est =  Residualize(model='GPR')
    elif name in ('augment', 'pad'):
        from confounds.base import Augment
        est =  Augment()
    elif name in ('dummy', 'passthrough'):
        from confounds.base import DummyDeconfounding
        est =  DummyDeconfounding()
    else:
        raise ValueError('Unrecognized model name! '
                         'Choose one of Residualize, Augment or Dummy.')

    return est


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


def check_list(input_combat):
    
    """ Check that combat data is a list and has 3 components"""
    
    # TODO: see sklearn if it's better using other than VAlueError here
    if type(input_combat) != list:
        raise ValueError(" Input data should be a list")
    if len(input_combat) != 3:
        raise ValueError(" Input data list should have three components")
        
    # Ensure 2D arrays of each componet   
    import numpy as np
    from sklearn.utils import check_array, column_or_1d
    
    input_combat[0] = check_array(input_combat[0], ensure_2d=False)
    input_combat[0] = column_or_1d(input_combat[0])
    
    if np.size(input_combat[1]) > 0:
        input_combat[1] = check_array(input_combat[1])
    if np.size(input_combat[2]) > 0:
        input_combat[2] = check_array(input_combat[2])
    
            
    # # Check that all have the same 
    # n_obs = np.array([elem.shape[1] for elem in input_combat if elem])
    # if np.any(n_obs != n_obs[0]):
    #     raise ValueError(" All input componets should have the same observations")
        
    #input_combat[0] = pd.get_dummies(input_combat[0])
        
    return input_combat
    