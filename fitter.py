from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from functools import partial

best = {}
best_error = float('inf')


def lgbm_objective(params, X_train, y_train, X_val, y_val, random_state):
    # the function gets a set of variable parameters in "params"
    global best, best_error
    params = {'n_estimators': int(params['n_estimators']),
              'max_depth': int(params['max_depth']),
              'learning_rate': params['learning_rate']}
    # we use this params to create a new LGBM Regressor
    model = LGBMRegressor(random_state=random_state, **params)
    model.fit(X_train, y_train)
    score = mean_absolute_error(y_val, model.predict(X_val))
    if score < best_error:
        best_error = score
        best = params
    return score


def hyperopt_lgbm_regression(X_train, y_train, X_val, y_val):
    # possible values of parameters
    space = {'n_estimators': hp.quniform('n_estimators', 20, 150, 10),
             'max_depth': hp.quniform('max_depth', 2, 10, 1),
             'learning_rate': hp.choice('learning_rate', [0.05, 0.1, 0.2])
             }
    # trials will contain logging information
    trials = Trials()
    n_iter = 20
    random_state = 42
    fmin_objective = partial(lgbm_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                             random_state=random_state)
    global best, best_error
    best = {}
    best_error = float('inf')
    fmin(fn=fmin_objective,  # function to optimize
         space=space,
         algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
         max_evals=n_iter,  # maximum number of iterations
         trials=trials,  # logging
         rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
         )
    # computing the score on the val set
    model = LGBMRegressor(random_state=random_state, n_estimators=int(best['n_estimators']),
                          max_depth=int(best['max_depth']), learning_rate=best['learning_rate'])

    model.fit(X_train, y_train)
    lgbm_model = [model, "lgbm", best]
    return lgbm_model


def rf_objective(params, X_train, y_train, X_val, y_val, random_state):
    # the function gets a set of variable parameters in "params"
    global best, best_error
    params = {'n_estimators': int(params['n_estimators']),
              'max_depth': int(params['max_depth']),
              'min_samples_leaf': params['min_samples_leaf'],
              'min_samples_split': params['min_samples_split']}
    # we use this params to create a new LGBM Regressor
    model = RandomForestRegressor(random_state=random_state, **params)
    model.fit(X_train, y_train)
    score = mean_absolute_error(y_val, model.predict(X_val))
    if score < best_error:
        best_error = score
        best = params
    return score


def hyperopt_rf_regression(X_train, y_train, X_val, y_val):
    # possible values of parameters
    space = {'n_estimators': hp.quniform('n_estimators', 20, 150, 10),
             'max_depth': hp.quniform('max_depth', 2, 10, 1),
             'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3, 4, 5]),
             'min_samples_split': hp.choice('min_samples_split', [2, 3, 4, 5, 6])
             }
    # trials will contain logging information
    trials = Trials()
    n_iter = 10
    random_state = 42
    fmin_objective = partial(rf_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                             random_state=random_state)
    global best, best_error
    best = {}
    best_error = float('inf')
    fmin(fn=fmin_objective,  # function to optimize
         space=space,
         algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
         max_evals=n_iter,  # maximum number of iterations
         trials=trials,  # logging
         rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
         )
    # computing the score on the val set
    model = RandomForestRegressor(random_state=random_state, n_estimators=int(best['n_estimators']),
                          max_depth=int(best['max_depth']), min_samples_leaf=best['min_samples_leaf'], min_samples_split=best['min_samples_split'])
    model.fit(X_train, y_train)
    rf_model = [model, "rf", best]
    return rf_model


def process_regression(X_train, y_train, X_val, y_val):
    lgbm_model = hyperopt_lgbm_regression(X_train, y_train, X_val, y_val)
    rf_model = hyperopt_rf_regression(X_train, y_train, X_val, y_val)
    candidates = [lgbm_model, rf_model]
    best_score = float('inf')
    for candidate in candidates:
        val_score = mean_absolute_error(y_val, candidate[0].predict(X_val))
        print(f'{candidate[1]} model val score {val_score}')
        if val_score < best_score:
            best_score = val_score
            best_model = candidate
    return best_model


def select_model(X_train, y_train, X_val, y_val, task):
    if task == 'regression':
        best_model = process_regression(X_train, y_train, X_val, y_val)
    else:
        return []
    return best_model
