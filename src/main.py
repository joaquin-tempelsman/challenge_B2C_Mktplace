import warnings
import logging
import os
import pandas as pd
from train_functions import (
    get_pipeline,
    split_and_process_data,
    objective,
    eval_model,
    get_optuna_plots,
    get_shap,
)
from preprocessing import unnest_data, preprocess
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import datetime
from datetime import datetime
from new_or_used import build_dataset
import joblib
from type_mapper import DataFrameDtypeMapper


# --PARAMS--#
SEED = None
SAVE_MODEL = False
CLF_task = True
RAW_DATA_PATH = "raw_data/MLA_100k_checked_v3.jsonlines"  # sys.argv[1]
TARGET = "condition"
SHAP = True
TRAIN_TEST_SPLIT = False
START_UP_TRIALS = 20  # int(sys.argv[2])
TRIALS = 70  # int(sys.argv[3])

warnings.simplefilter("ignore")

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


cat_bool_features = [
    "currency_id",
    "video_id",
    "automatic_relist",
    "shippinglocal_pick_up",
    "shippingfree_shipping",
    "nmp_cash",
    "nmp_giro_postal",
    "nmp_buyer",
    "nmp_transf",
    "nmp_tc",
    "warranty",
    "tag_good_qt",
    "tag_dragg_bv",
    "accepts_mercadopago",
    "wknd_hlday_created",
    "wknd_hlday_start_time",
    "wknd_hlday_stop_time",
]


cat_id_features = [
    "seller_id",
    "listing_type_id",
    # "parent_item_id", #same as category_id
    "category_id",
    "seller_addressstate_id",
    "seller_addresscity_id",
]

count_enc_features = ["seller_id_cnt", "category_id_cnt"]

cat_base_features = [
    "buying_mode",
    "status",
    "shippingmode",
    "seller_addressstate_name",
]

num_features = [
    "price",
    "initial_quantity",
    "sold_quantity",
    "available_quantity",
    "pic_num_items",
    "pic_max_size",
    "pic_min_size",
    "pic_num_sizes",
    "nmp_qty",
    "days_since_update",
    "days_elapsed",
]

drop_after_encoding_features = ["date_created"]

# types casting for lgbm
cols_to_bool = cat_bool_features
cols_to_float = cat_id_features + num_features + count_enc_features
cols_to_cat = cat_base_features

all_features = (
    cat_bool_features
    + cat_id_features
    + cat_base_features
    + num_features
    + count_enc_features
    + drop_after_encoding_features
)

if __name__ == "__main__":
    run_time = datetime.now()
    output_path = "src/trained_models/" + datetime.strftime(run_time, "%Y-%m-%d_%H%M%S")

    logging.info(f"creating output folder at {output_path}")
    os.mkdir(output_path)

    logging.info(f"reading raw data from: {RAW_DATA_PATH}")
    X_train, y_train, X_test, y_test = build_dataset(RAW_DATA_PATH)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    X_train = unnest_data(X_train)
    X_test = unnest_data(X_test)

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    X_train = X_train.loc[:, all_features]
    X_test = X_test.loc[:, all_features]

    y_train = pd.Series([1 if item == "used" else 0 for item in y_train])
    y_test = pd.Series([1 if item == "used" else 0 for item in y_test])

    TypeMapper = DataFrameDtypeMapper(
        X_train.columns, cols_to_bool, cols_to_float, cols_to_cat
    )

    X_train[[x for x in all_features if x not in num_features]] = X_train[
        [x for x in all_features if x not in num_features]
    ].astype("category")

    if TRAIN_TEST_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.10, random_state=SEED, shuffle=True
        )

    logging.info("preparing processed train/val folds")
    (
        X_training_folds,
        y_training_folds,
        X_testing_folds,
        y_testing_folds,
    ) = split_and_process_data(
        X_train,
        y_train,
        get_pipeline(
            numerical_cols=num_features,
            categorical_cols=[
                x
                for x in all_features
                if x not in num_features + drop_after_encoding_features
            ],
            input_missing=cat_base_features + count_enc_features,
            cbe_enc=cat_id_features,
            rl_enc=cat_base_features,
            count_enc=count_enc_features,
        ),
        KFold(n_splits=5, random_state=SEED, shuffle=True),
        TypeMapper=TypeMapper,
        drop_after_encoding_features=drop_after_encoding_features,
    )

    logging.info(
        f"study definition with {START_UP_TRIALS} start up trials and {TRIALS} trials"
    )
    study = optuna.create_study(
        direction="maximize" if CLF_task else "minimize",
        sampler=TPESampler(n_startup_trials=START_UP_TRIALS),
    )

    logging.info("optuna study start")
    study.optimize(
        lambda trial: objective(
            trial,
            X_training_folds,
            X_testing_folds,
            y_training_folds,
            y_testing_folds,
            clf=CLF_task,
            seed=SEED,
        ),
        n_trials=TRIALS,
        n_jobs=-1,
    )

    logging.info("optuna study end")

    logging.info("dump optuna study")
    joblib.dump(study, f"{output_path}/study.joblib")

    pipeline_test = get_pipeline(
        numerical_cols=num_features,
        categorical_cols=[x for x in all_features if x not in num_features],
        input_missing=cat_base_features + count_enc_features,
        cbe_enc=cat_id_features,
        rl_enc=cat_base_features,
        count_enc=count_enc_features,
    )

    logging.info("evaluating model on OOS data")
    X_train_transformed = pipeline_test.fit_transform(X_train, y_train)

    X_train_transformed = TypeMapper.map_col_names(X_train_transformed)

    X_train_transformed = TypeMapper.cast_type(X_train_transformed)

    if CLF_task:
        model = LGBMClassifier(**study.best_params, random_state=SEED)
    else:
        model = LGBMRegressor(**study.best_params, random_state=SEED)

    model = model.fit(X_train_transformed, y_train)
    X_test_transformed = pipeline_test.transform(X_test)

    X_test_transformed = TypeMapper.map_col_names(X_test_transformed)

    X_test_transformed = TypeMapper.cast_type(X_test_transformed)

    eval_model(y_test, X_test_transformed, model, output_path, clf=CLF_task)

    if SHAP:
        get_shap(model, X_train_transformed, output_path)

    # ! put after study end
    logging.info("saving optuna metrics")
    get_optuna_plots(study, output_path)

    if SAVE_MODEL:
        logging.info("fitting best model to all data")
        logging.info("building pipeline")

        pipeline_test = get_pipeline(
            numerical_cols=num_features,
            categorical_cols=[x for x in all_features if x not in num_features],
            input_missing=cat_base_features + count_enc_features,
            cbe_enc=cat_id_features,
            rl_enc=cat_base_features,
            count_enc=count_enc_features,
        )

        # # ! FIX - concat all the data before this step
        # logging.info("fit transform pipeline")
        # Transformed_X = pipeline.fit_transform(X, y).astype(float)

        # logging.info("fit optimized model with all data")
        # full_data_model = LGBMClassifier(**study.best_params).fit(Transformed_X, y)

        # # logging.info("saving feature importance")
        # # feat_importance_df = pd.DataFrame(
        # #     {
        # #         "importance": full_data_model.feature_importances_,
        # #         "feature": numerical_features + categorical_features,
        # #     }
        # # ).sort_values(by="importance", ascending=False)
        # # feat_importance_df.to_csv(f"{output_path}/feat_importance.csv")

        # logging.info("dump trained model")
        # joblib.dump(full_data_model, f"{output_path}/model.joblib")

        # logging.info("dump trained model")
        # joblib.dump(pipeline, f"{output_path}/pipeline.joblib")

    time_elapsed = datetime.now() - run_time
    logging.info(f"program ended correctly - time elapsed: {time_elapsed}")
