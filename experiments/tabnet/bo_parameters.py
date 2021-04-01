from skopt.space import Real, Integer, Categorical

SEARCH_SPACE_LGBM = [
    Real(0.1, 0.5, name='learning_rate'),
    Integer(3, 15, name='max_depth'),
    Integer(100, 1000, name='n_estimators'),
]

SEARCH_SPACE_XGB = [
    Real(0.1, 1.0, name='learning_rate'),
    Integer(3, 15, name='max_depth'),
    Integer(100, 1000, name='n_estimators'),
]

SEARCH_SPACE_TABNET = [
    # Real(0.01, 1.0, name='lr'),
    Real(1.0, 2.0, name='gamma'),
    Real(0.0001, 0.1, name='lambda_sparse'),
    Integer(3, 10, name='n_steps'),
    Categorical([8, 16, 24, 32, 64], name='n_a'),
    Categorical([0.6, 0.7, 0.8, 0.9, 0.95, 0.98], name='momentum'),
    Categorical([4096, 8192], name='batch_size'),
    Categorical([128, 256], name='virtual_batch_size'),
]

LIGHTGBM_PARAMS = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'silent': True,
}

XGBOOST_PARAMS = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42,
        'silent': True,
}

TABNET_PARAMS = {
        'seed': 42,
        'verbose': True,
}