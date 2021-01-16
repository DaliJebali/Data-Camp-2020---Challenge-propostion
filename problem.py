import os
import pandas as pd
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit


problem_title = 'Data Science Business Case: Churn Prediction'
_target_column_name = 'CHURNED'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names = [0, 1])

# An object implementing the workflow
workflow = rw.workflows.Estimator()

#We will use AUC-ROC score to assess the models
score_types = [
    rw.score_types.ROCAUC(name='AUC-ROC_score', precision=5),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), index_col='CUSTOMER_ID')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return pd.DataFrame(X_df), y_array


def get_train_data(path='.'):
    f_name = 'churn_train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'churn_test.csv'
    return _read_data(path, f_name)
