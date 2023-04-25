## Challenge @ Mercado Libre


In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

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
| seller_id                |             1.878409387 |
| listing_type_id          |             1.396018496 |
| category_id              |            0.7676490883 |
| sold_quantity            |            0.6142852918 |
| initial_quantity         |            0.3223816614 |
| available_quantity       |            0.2342971204 |
| pic_num_sizes            |            0.2184315057 |
| seller_antiq             |            0.1632172264 |
| pic_min_size             |             0.148567578 |
| pic_num_items            |            0.1331583003 |
| price                    |            0.1280903114 |
| pic_max_size             |            0.1259008054 |
| nmp_tc                   |            0.1133652086 |
| seller_addresscity_id    |           0.08137914146 |
| shippingmode             |            0.0803226149 |
| tag_good_qt              |           0.07169756723 |
| nmp_qty                  |           0.05354144618 |
| nmp_buyer                |           0.04679937248 |
| seller_addressstate_name |           0.03537822569 |
| days_since_update        |           0.03376469802 |
| buying_mode              |           0.03090938423 |
| days_elapsed             |           0.02982233685 |
| nmp_transf               |           0.02194221676 |
| category_id_cnt          |           0.02002994346 |
| automatic_relist         |           0.01738736192 |