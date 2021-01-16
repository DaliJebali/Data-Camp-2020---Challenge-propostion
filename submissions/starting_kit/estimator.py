#import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, \
    OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def _deal_with_missing_values(X):
    X = X.replace([np.inf, -np.inf], np.nan)  # Here we change infinite values to NaN
    X = X.fillna(-999)
    return X


def _preprocess_categorical_features(X):
    """Preprocess step"""
    X['COLLEGE'] = np.where((X.COLLEGE == 'one'), 1, X.COLLEGE)
    X['COLLEGE'] = np.where((X.COLLEGE == 'zero'), 0, X.COLLEGE)

    X['LESSTHAN600k'] = np.where((X.LESSTHAN600k == True), 1, X.LESSTHAN600k)
    X['LESSTHAN600k'] = np.where((X.LESSTHAN600k == False), 0, X.LESSTHAN600k)

    return X #USEFUL if there are categorical features


transformer_missing_values = FunctionTransformer(
    lambda X_df: _deal_with_missing_values(X_df)
)

transformer_cat = FunctionTransformer(
    lambda X_df: np.array(_preprocess_categorical_features(X_df))
)

cols = ['DATA', 'INCOME', 'OVERCHARGE', 'LEFTOVER', 'HOUSE',
        'CHILD', 'JOB_CLASS', 'REVENUE', 'HANDSET_PRICE',
        'OVER_15MINS_CALLS_PER_MONTH', 'TIME_CLIENT', 'AVERAGE_CALL_DURATION',
        'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL',
        'CONSIDERING_CHANGE_OF_PLAN']

categorical_binary = ['COLLEGE', 'LESSTHAN600k']

transformer = make_column_transformer(
    (transformer_cat, categorical_binary),
    (transformer_missing_values, cols + categorical_binary),
    ('passthrough', cols)
)

pipe= make_pipeline(
    transformer,
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
    RandomForestClassifier()
    )


def get_estimator():
    return pipe
