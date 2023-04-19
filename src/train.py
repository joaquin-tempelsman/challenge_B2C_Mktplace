import warnings
warnings.simplefilter("ignore")
import sys
import logging
import joblib
import os
import pandas as pd
from train_functions import (get_pipeline, split_and_process_data, objective,
                            evaluate_regression, get_optuna_plots)
from data_prep import get_preproccessed_data
from preprocessing_funcs import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
import datetime
from datetime import datetime

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

raw_data_folder = sys.argv[1]
start_up_trials = int(sys.argv[2])
trials = int(sys.argv[3])

if __name__ == '__main__':
    
    #control randomness
    seed = None
    
    run_time = datetime.now()
    output_path = 'trained_models/' + datetime.strftime(run_time, "%Y-%m-%d_%H%M%S")
    
    logging.info(f'creating output folder at {output_path}')
    os.mkdir(output_path)
    
    logging.info(f'reading files data from: {raw_data_folder}')
    df = get_preproccessed_data(raw_data_folder)
    df = df.loc[df.car_parking_address_city == 'San Francisco']

    ### -- FOR TESTING PURPOSES - REMOVE COMMENT IF NOT TESTING -- ###
    #df = df.sample(frac=0.3)
    ### -- FOR TESTING PURPOSES - REMOVE COMMENT IF NOT TESTING -- ###


    target_col = 'trip_sum_trip_price' 
    
    numerical_features = [
        'reservation_lead_time',
        'hist_usage_median',
        'hist_usage_q10',
        'hist_free_cars',
        'trip_duration'
    ]

    categorical_features = [
        'weekend',
        'car_parking_address_postcode',
        '7_to_12hs',
        'car_id_hash'
        ]


    logging.info(f'Casting cat features to category') 
    df[categorical_features] = df[categorical_features].astype('category')
    
    X = df[categorical_features + numerical_features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.10, random_state=seed, shuffle=True)


    logging.info('preparing processed train/test folds')
    X_training_folds, y_training_folds, X_testing_folds, y_testing_folds = (
        split_and_process_data( X_train, 
                                y_train, 
                                get_pipeline(categorical_features,
                                             numerical_features), 
                                             KFold(n_splits=5, 
                                                   random_state = seed, 
                                                   shuffle=True))
                            )
    
    #default set at 20 start up / 70 trials
    logging.info(f'study definition with {start_up_trials} start up trials and {trials} trials')
    study = optuna.create_study(direction = 'minimize',
                                sampler=TPESampler(
                                n_startup_trials = start_up_trials))  
    
    logging.info('optuna study start')
    study.optimize(lambda trial: objective(trial, 
                                            X_training_folds, 
                                            X_testing_folds, 
                                            y_training_folds, y_testing_folds), 
                                            n_trials=trials, n_jobs=-1)
    
    logging.info('optuna study end')

    logging.info('saving optuna metrics')
    get_optuna_plots(study,f'{output_path}')
    
    logging.info('evaluating model on OOS data')
    eval_model = LGBMRegressor(**study.best_params).fit(X_train, y_train)
    evaluate_regression(y_test, eval_model.predict(X_test), output_path)

    logging.info('fitting best model to all data')
    logging.info('building pipeline')
    pipeline = get_pipeline(categorical_features, numerical_features)

    logging.info('fit transform pipeline')
    Transformed_X = pipeline.fit_transform(X, y)

    logging.info('fit optimized model with all data')
    full_data_model = LGBMRegressor(**study.best_params).fit(Transformed_X, y)

    logging.info('saving feature importance')
    feat_importance_df = pd.DataFrame({'importance': full_data_model.feature_importances_, 'feature': numerical_features + categorical_features}).sort_values(by='importance',ascending=False)
    feat_importance_df.to_csv(f'{output_path}/feat_importance.csv')
    
    logging.info('dump trained model')
    joblib.dump(full_data_model , f'{output_path}/model.joblib')

    logging.info('dump trained model')
    joblib.dump(pipeline , f'{output_path}/pipeline.joblib')
    
    logging.info('dump optuna study')
    joblib.dump(study , f'{output_path}/study.joblib')

    time_elapsed = datetime.now() - run_time 
    logging.info(f'program ended correctly - time elapsed: {time_elapsed}')