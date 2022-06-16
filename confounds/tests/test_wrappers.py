import numpy as np

from confounds import Residualize, Augment, DummyDeconfounding
from confounds.sklearn import (DeconfEstimator, deconfounded_cv_predict,
                               deconfounded_cv_score)

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression


def test_estimator_regression():

    from sklearn.datasets import make_regression

    regressor = LinearRegression()
    X, y = make_regression(n_features=10)

    C = X[:, 7:]
    X = X[:, :7]

    X_train, X_test, y_train, y_test, C_train, C_test = \
        train_test_split(X, y, C, random_state=1234)

    for deconf in [Residualize(), Augment(), DummyDeconfounding()]:
        deconf.fit(X_train, C_train)
        X_train_deconf = deconf.transform(X_train, C_train)
        X_test_deconf = deconf.transform(X_test, C_test)

        regressor.fit(X_train_deconf, y_train)
        y_pred = regressor.predict(X_test_deconf)
        score = regressor.score(X_test_deconf, y_test)

        deconf_estim = DeconfEstimator(deconfounder=deconf,
                                       estimator=regressor)

        deconf_estim.fit(X_train, y_train, confounders=C_train)
        assert np.allclose(y_pred,
                           deconf_estim.predict(X_test, confounders=C_test)
                           )

        assert np.allclose(score,
                           deconf_estim.score(X_test, y_test,
                                              confounders=C_test)
                           )


def test_estimator_classification():

    from sklearn.datasets import make_classification

    clf = LogisticRegression()
    X, y = make_classification(n_features=10)

    C = X[:, 7:]
    X = X[:, :7]

    X_train, X_test, y_train, y_test, C_train, C_test = \
        train_test_split(X, y, C, random_state=1234)

    for deconf in [Residualize(), Augment(), DummyDeconfounding()]:
        deconf.fit(X_train, C_train)
        X_train_deconf = deconf.transform(X_train, C_train)
        X_test_deconf = deconf.transform(X_test, C_test)

        clf.fit(X_train_deconf, y_train)
        y_pred = clf.predict(X_test_deconf)
        score = clf.score(X_test_deconf, y_test)

        deconf_estim = DeconfEstimator(deconfounder=deconf,
                                       estimator=clf)

        deconf_estim.fit(X_train, y_train, confounders=C_train)
        assert np.allclose(y_pred,
                           deconf_estim.predict(X_test, confounders=C_test)
                           )

        assert np.allclose(score,
                           deconf_estim.score(X_test, y_test,
                                              confounders=C_test)
                           )


def test_cv_regression():

    from sklearn.datasets import make_regression

    X, y = make_regression(n_features=10)

    C = X[:, 7:]
    X = X[:, :7]

    regressor = LinearRegression()
    deconf = Residualize()
    deconf_estim = DeconfEstimator(deconf, regressor)

    cv = KFold(n_splits=5, shuffle=False)

    scores = []
    predictions = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        C_train, C_test = C[train_index], C[test_index]

        deconf.fit(X_train, C_train)
        X_train_deconf = deconf.transform(X_train, C_train)
        X_test_deconf = deconf.transform(X_test, C_test)

        regressor.fit(X_train_deconf, y_train)
        predictions.append(regressor.predict(X_test_deconf))
        scores.append(regressor.score(X_test_deconf, y_test))

    predictions = np.concatenate(predictions)

    assert np.allclose(predictions,
                       deconfounded_cv_predict(deconf_estim, X, y,
                                               confounds=C, cv=cv)
                       )

    assert np.allclose(scores,
                       deconfounded_cv_score(deconf_estim, X, y,
                                             confounds=C, cv=cv)
                       )


def test_cv_classification():

    from sklearn.datasets import make_classification

    # Do classification problem
    X, y = make_classification(n_features=10)

    C = X[:, 7:]
    X = X[:, :7]

    clf = LogisticRegression()
    deconf = Residualize()
    deconf_estim = DeconfEstimator(deconf, clf)

    cv = KFold(n_splits=5, shuffle=False)

    scores = []
    predictions = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        C_train, C_test = C[train_index], C[test_index]

        deconf.fit(X_train, C_train)
        X_train_deconf = deconf.transform(X_train, C_train)
        X_test_deconf = deconf.transform(X_test, C_test)

        clf.fit(X_train_deconf, y_train)
        predictions.append(clf.predict(X_test_deconf))
        scores.append(clf.score(X_test_deconf, y_test))

    predictions = np.concatenate(predictions)

    assert np.allclose(predictions,
                       deconfounded_cv_predict(deconf_estim, X, y,
                                               confounds=C, cv=cv)
                       )

    assert np.allclose(scores,
                       deconfounded_cv_score(deconf_estim, X, y,
                                             confounds=C, cv=cv)
                       )
