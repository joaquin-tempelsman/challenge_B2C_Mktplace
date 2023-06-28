## Challenge @ Marketplace


In the context of a B2C amazon like Marketplace, an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb

##  Results

Experiment `2023-04-24_224335`  
Test metrics:   
accuracy: 0.9185  
auc_pr: 0.8642  
auc: 0.9762  

number of features: 25

| feature                  | shap_feature_importance |
|--------------------------|-------------------------|
| seller_id                |             1.8784 |
| listing_type_id          |             1.3960 |
| category_id              |            0.7676 |
| sold_quantity            |            0.6142 |
| initial_quantity         |            0.3223 |
| available_quantity       |            0.2342 |
| pic_num_sizes            |            0.2184 |
| seller_antiq             |            0.1632 |
| pic_min_size             |             0.1485 |
| pic_num_items            |            0.1331 |
| price                    |            0.1280 |
| pic_max_size             |            0.1259 |
| nmp_tc                   |            0.1133 |
| seller_addresscity_id    |           0.0813 |
| shippingmode             |            0.0803 |
| tag_good_qt              |           0.0716 |
| nmp_qty                  |           0.0535 |
| nmp_buyer                |           0.0467 |
| seller_addressstate_name |           0.0353 |
| days_since_update        |           0.0337 |
| buying_mode              |           0.0309 |
| days_elapsed             |           0.0298 |
| nmp_transf               |           0.0219 |
| category_id_cnt          |           0.0200 |
| automatic_relist         |           0.0173 |

## Training logs  
40 startup trials + 140 trials  
<img src=src/trained_models/2023-04-24_224335/param_optimization_history.png alt= “asd” width="450" height="321">
<img src=src/trained_models/2023-04-24_224335/param_optimization_importance.png alt= “asd” width="450" height="321">
  
  
- boosting_type: 'gbdt'  
- metric: 'binary_logloss'
- n_estimators: 625
- learning_rate: 0.10240196706281192
- num_leaves: 2907
- max_depth: 9
- min_child_samples: 453
- reg_alpha: 3
- reg_lambda: 81
- min_split_gain: 0.056866046171603846
- subsample: 0.5
- colsample_bytree: 0.9000000000000001
## Next steps  
- perform partial dependance plots / shap beeswarm plots to understand the model explainability for top 10 most important features at least to validate the model
- add features that describe how many post the seller made in the last N days (multiple windows)  
- add 'sigma' parameter from CatBoostEncoder to the search space of the hyperparameter tunning search space as well as other relevante parameters that can impact the features encoding step
- multi-hot encode the title (after cleaning noise), train a Deep learning simple model to predict the probability for the title to belong to a 'used' item and use that probability as a feature
- modify categorical feature pipeline to use OrdinalEncoder first so it handles unknown categories correctly during inference time and then pass them to the 'categorical_feature' param in LGBMClassifier object instead of using .astype('category')
- build a feature using 'title' column to generate a probability (target encoded variable) using DL (test with 2 layers at first + sigmoid, could be enough)
- modify pipeline so the metrics to optimize in the trial for classification have two settings, one for is_imbalance == True (using AUC_ROC, AUC_PR and ks score, and for balanced datasets binary logloss)



