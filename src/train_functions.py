

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.cat_boost import CatBoostEncoder
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.selection import DropConstantFeatures
import numpy as np
from lightgbm import LGBMRegressor
from optuna import Trial
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import optuna
import plotly.io as pio
import matplotlib.pyplot as plt
from numpy import sqrt 
import json
import statsmodels.api as sm

def get_categorical_pipeline(categorical_cols):
    
    pipeline = Pipeline([
    ('categorical_imputer', CategoricalImputer(
        imputation_method='missing', 
        variables=categorical_cols,
        return_object=True
    )),
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05,
        variables=categorical_cols,
        ignore_format=True
    )),
    ('cat_boost_enc', CatBoostEncoder(cols=categorical_cols))
    ])
    
    return pipeline



def get_numerical_pipeline(numerical_cols):
    
    pipeline = Pipeline([
    ('numerical_imputer', ArbitraryNumberImputer(
        arbitrary_number=10**8, 
        variables=numerical_cols
    ))
    ])
    
    return pipeline



def get_pipeline(categorical_cols, numerical_cols, model=None):
    
    if numerical_cols: 
        numerical_pipeline = get_numerical_pipeline(numerical_cols)
    
    if categorical_cols:
        categorical_pipeline = get_categorical_pipeline(categorical_cols)
    
    pipeline_tail = [('selector', DropConstantFeatures(tol=0.995))]+([('model', model)] if model else [])
    
    if numerical_cols and categorical_cols:
        pipeline = Pipeline([
            ('transformer', ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_cols),
                ('categorical_pipeline', categorical_pipeline, categorical_cols)
            ])),            
        ] + pipeline_tail)
    
    elif numerical_cols:
        pipeline = Pipeline([
            ('transformer', ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_cols),
            ])),
        ] + pipeline_tail)
        
    else: 
        pipeline = Pipeline([
            ('transformer', ColumnTransformer([
                ('categorical_pipeline', categorical_pipeline, categorical_cols),
            ])),
        ] + pipeline_tail)
    
    return pipeline


def split_and_process_data(X_train, y_train, pipeline, kfold):

    X_training_folds = []
    y_training_folds = []
    X_test_folds = []
    y_test_folds = []

    for train_ix, test_ix in kfold.split(X_train, y_train):
        fold_X_train, fold_X_test = X_train.iloc[train_ix], X_train.iloc[test_ix]
        fold_y_train, fold_y_test = y_train.iloc[train_ix], y_train.iloc[test_ix]

        X_transformed = pipeline.fit_transform(fold_X_train, fold_y_train)
        X_transformed_test = pipeline.transform(fold_X_test)

        X_training_folds.append(X_transformed)
        y_training_folds.append(fold_y_train)
        X_test_folds.append(X_transformed_test)
        y_test_folds.append(fold_y_test)
    
    return X_training_folds, y_training_folds, X_test_folds, y_test_folds

def instantiate_lgbm(trial : Trial) -> LGBMRegressor:
    params_dict : Dict[str, Any] = {
        #"device_type": trial.suggest_categorical("device_type", ['gpu']),
        "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt ', 'dart']),
        #"metric": trial.suggest_categorical("metric", ['auc', 'average_precision', 'binary_logloss']),
        "metric": trial.suggest_categorical("metric", ['rmse','mape']),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),  #  (1e-4,1e-5) (0.2,0.1)
        "num_leaves": trial.suggest_int("num_leaves", 7, 4095, step=20), 
        "max_depth": trial.suggest_int("max_depth", 2, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 200, 10000),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
        "subsample": trial.suggest_float( "subsample", 0.4, 1, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1, step=0.05),
        "random_state": 7
    }
    
    return LGBMRegressor(**params_dict)


def objective(trial, X_training_folds, X_testing_folds, y_training_folds, y_testing_folds):
    lgbm = instantiate_lgbm(trial)
    results = ([mean_squared_error(y_test, lgbm.fit(X_train, y_train).predict(X_test), 
                                   squared=False) for X_train, X_test, y_train, y_test in 
                                            zip(X_training_folds,
                                                X_testing_folds,
                                                y_training_folds,
                                                y_testing_folds)])
    
    return np.mean(results)



def evaluate_regression(y_test, y_pred, save_path):

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    metrics = {'r2': r2, 'rmse': rmse}
    with open(f'{save_path}/test_metrics.json', 'w') as outfile:
            json.dump(metrics, outfile)
    print('test_metrics r2 - rmse: ',metrics)
    fig = sm.qqplot(y_test - y_pred)
    plt.savefig(f'{save_path}/qq_plot.png')

    plt.hist(y_test - y_pred, bins= 10)
    plt.savefig(f'{save_path}/resid_hist_plot.png')

def get_optuna_plots(study,save_path):

    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image(f'{save_path}/param_optimization_history.png')
    
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image(f'{save_path}/param_optimization_importance.png')


    
    
