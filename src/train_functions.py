from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from category_encoders.cat_boost import CatBoostEncoder
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from optuna import Trial
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
import optuna
import plotly.io as pio
import matplotlib.pyplot as plt
from numpy import sqrt
import json
import statsmodels.api as sm
from typing import Dict, Union, Any
import shap
import pandas as pd
from encoding import encode_seller_antiq


def get_pipeline(
    numerical_cols, categorical_cols, input_missing, cbe_enc, rl_enc, count_enc
):
    # Define the transformers
    cat_transformer = make_pipeline(
        CategoricalImputer(
            imputation_method="missing",
            variables=input_missing,
            return_object=True,
        ),
        RareLabelEncoder(
            tol=0.01, variables=rl_enc, ignore_format=True, n_categories=0
        ),
        CountFrequencyEncoder(variables=count_enc, ignore_format=True),
        CatBoostEncoder(cols=cbe_enc),
    )

    num_transformer = make_pipeline(
        ArbitraryNumberImputer(arbitrary_number=10**8, variables=numerical_cols)
    )

    # Combine the transformers into a single pipeline
    preprocessor = make_column_transformer(
        (cat_transformer, categorical_cols),
        (num_transformer, numerical_cols),
        remainder="passthrough",
    )

    # Define the full pipeline
    pipeline = make_pipeline(
        preprocessor,
        # Add additional transformers or estimators here as needed
    )

    return pipeline


def split_and_process_data(
    X_train, y_train, pipeline, kfold, TypeMapper, drop_after_encoding_features
):
    X_training_folds = []
    y_training_folds = []
    X_test_folds = []
    y_test_folds = []

    for train_ix, test_ix in kfold.split(X_train, y_train):
        fold_X_train, fold_X_test = X_train.iloc[train_ix], X_train.iloc[test_ix]
        fold_y_train, fold_y_test = y_train.iloc[train_ix], y_train.iloc[test_ix]

        fold_X_train, fold_X_test = apply_feature_transform_pipeline(
            fold_X_train,
            fold_y_train,
            fold_X_test,
            pipeline,
            TypeMapper,
            drop_after_encoding_features,
        )

        X_training_folds.append(fold_X_train)
        y_training_folds.append(fold_y_train)
        X_test_folds.append(fold_X_test)
        y_test_folds.append(fold_y_test)

    return X_training_folds, y_training_folds, X_test_folds, y_test_folds


def instantiate_lgbm(
    trial: Trial, clf=True, seed=None
) -> Union[LGBMRegressor, LGBMClassifier]:
    if clf:
        params_dict: Dict[str, Any] = {
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt ", "dart"]
            ),
            "metric": trial.suggest_categorical(
                "metric", ["auc", "average_precision", "binary_logloss"]
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.5, log=True
            ),  #  (1e-4,1e-5) (0.2,0.1)
            "num_leaves": trial.suggest_int("num_leaves", 7, 4095, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 200, 10000),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float("subsample", 0.4, 1, step=0.05),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 1, step=0.05
            ),
            "random_state": seed,
        }

        return LGBMClassifier(**params_dict)

    else:
        params_dict: Dict[str, Any] = {
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt ", "dart"]
            ),
            "metric": trial.suggest_categorical("metric", ["rmse", "mape"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.5, log=True
            ),  #  (1e-4,1e-5) (0.2,0.1)
            "num_leaves": trial.suggest_int("num_leaves", 7, 4095, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 200, 10000),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 15),
            "subsample": trial.suggest_float("subsample", 0.4, 1, step=0.05),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 1, step=0.05
            ),
            "random_state": 7,
        }

        return LGBMRegressor(**params_dict)


def objective(
    trial,
    X_training_folds,
    X_testing_folds,
    y_training_folds,
    y_testing_folds,
    clf=True,
    seed=None,
):
    lgbm = instantiate_lgbm(trial, clf=clf, seed=seed)

    if clf:
        results = [
            accuracy_score(y_test, lgbm.fit(X_train, y_train).predict(X_test))
            for X_train, X_test, y_train, y_test in zip(
                X_training_folds, X_testing_folds, y_training_folds, y_testing_folds
            )
        ]
    else:
        results = [
            mean_squared_error(
                y_test, lgbm.fit(X_train, y_train).predict(X_test), squared=False
            )
            for X_train, X_test, y_train, y_test in zip(
                X_training_folds, X_testing_folds, y_training_folds, y_testing_folds
            )
        ]

    return np.mean(results)


def eval_model(y_test, X_test, test_clf, save_path, clf):
    y_pred = test_clf.predict(X_test)
    if clf:
        y_pred_proba = test_clf.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred)
        metrics = {"accuracy": accuracy, "auc_pr": auc_pr, "auc": auc}

    else:
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics = {"r2": r2, "rmse": rmse}

        print("test_metrics r2 - rmse: ", metrics)
        fig = sm.qqplot(y_test - y_pred)
        plt.savefig(f"{save_path}/qq_plot.png")

        plt.hist(y_test - y_pred, bins=10)
        plt.savefig(f"{save_path}/resid_hist_plot.png")

    with open(f"{save_path}/test_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)


def get_optuna_plots(study, save_path):
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image(f"{save_path}/param_optimization_history.png")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image(f"{save_path}/param_optimization_importance.png")


def get_shap(model, X_train_transformed, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_transformed)

    feature_names = X_train_transformed.columns

    rf_resultX = pd.DataFrame(shap_values[0], columns=feature_names)

    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)), columns=["col_name", "feature_importance_vals"]
    )
    shap_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )

    shap_importance.to_csv(f"{output_path}/shap_importance.csv", index=False)


def apply_feature_transform_pipeline(
    X_train, y_train, X_test, pipeline, TypeMapper, drop_after_encoding_features
):
    X_train = pipeline.fit_transform(X_train, y_train)
    X_test = pipeline.transform(X_test)

    X_train = TypeMapper.map_col_names(X_train)
    X_test = TypeMapper.map_col_names(X_test)

    # cast types after pipeline transform
    X_train = TypeMapper.cast_type(
        X_train,
    )

    X_test = TypeMapper.cast_type(
        X_test,
    )

    # - custom encoding -#
    X_train, X_test = encode_seller_antiq(X_train, X_test)
    X_train.drop(drop_after_encoding_features, axis=1, inplace=True)
    X_test.drop(drop_after_encoding_features, axis=1, inplace=True)

    return X_train, X_test
