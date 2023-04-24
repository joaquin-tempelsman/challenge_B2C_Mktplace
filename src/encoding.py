import pandas as pd


def get_time_difference(seller_id, date_created, seller_id_first_pub):
    first_post = seller_id_first_pub.get(seller_id)
    if first_post:
        seller_antiq = (date_created - first_post).days
    else:
        seller_antiq = None

    return seller_antiq


def encode_seller_antiq(X_train, X_test):
    X_train["date_created"] = pd.to_datetime(X_train["date_created"])
    X_test["date_created"] = pd.to_datetime(X_test["date_created"])

    seller_id_first_pub: dict = (
        X_train.groupby("seller_id")["date_created"].min().to_dict()
    )
    X_train["seller_antiq"] = X_train.apply(
        lambda x: get_time_difference(
            x["seller_id"], x["date_created"], seller_id_first_pub
        ),
        axis=1,
    )
    X_test["seller_antiq"] = X_test.apply(
        lambda x: get_time_difference(
            x["seller_id"], x["date_created"], seller_id_first_pub
        ),
        axis=1,
    )
    return X_train, X_test
