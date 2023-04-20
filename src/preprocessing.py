import pandas as pd
from pandas import DataFrame
import re
import numpy as np
import holidays
import string


def unnest_data(df) -> DataFrame:
    df = unnest_dict_columns(df, "seller_address")
    df = unnest_dict_columns(df, "shipping")
    df["sub_status"] = df["sub_status"].apply(unpack_list)
    df["deal_ids"] = df["deal_ids"].apply(unpack_list)
    df["shippingtags"] = df["shippingtags"].apply(unpack_list)
    df["descriptions"] = df["descriptions"].apply(parse_descriptions)

    unpacked_cols = [
        "seller_address",
        "shipping",
        "descriptions",
        "sub_status",
        "deal_ids",
        "shippingtags",
    ]

    df.drop(columns=unpacked_cols, inplace=True)

    return df


def preprocess(df) -> DataFrame:
    df[["pic_num_items", "pic_max_size", "pic_min_size", "pic_num_sizes"]] = df.apply(
        lambda x: extract_pic_features(x["pictures"]), axis=1, result_type="expand"
    )

    df[
        [
            "nmp_cash",
            "nmp_giro_postal",
            "nmp_buyer",
            "nmp_mp",
            "nmp_transf",
            "nmp_tc",
            "nmp_qty",
        ]
    ] = df.apply(
        lambda x: extract_nmp_payment_methods_features(
            x["non_mercado_pago_payment_methods"]
        ),
        axis=1,
        result_type="expand",
    )

    df[
        [
            "tag_good_qt",
            "tag_poor_qt",
            "tag_dragg_v",
            "tag_dragg_bv",
        ]
    ] = df.apply(
        lambda x: extract_tag_features(x["tags"]),
        axis=1,
        result_type="expand",
    )

    df["warranty"] = df["warranty"].apply(get_warranty_features)

    df["total_qty"] = df.apply(
        lambda x: get_variations_feature(x["variations"]), axis=1
    )

    df["seller_addresscity_id"] = np.where(
        df.seller_addresscity_id == "", None, df.seller_addresscity_id
    )

    df["status"] = np.where(
        df["status"] == "not_yet_active", None, df["status"]
    )  # fix cardinality issue

    df["title_used"] = df["title"].str.lower().str.contains("usado|used")

    df["video_id"] = df["video_id"].isna()

    df["currency_id"] = np.where(df["currency_id"] == "ARS", True, False)

    df["thumbnail"] = np.where(df["thumbnail"].apply(len) == 0, False, True)

    df["thumbnail_diff"] = df["secure_thumbnail"] != df["thumbnail"]

    df["shippingfree_methods"] = df["shippingfree_methods"].isna()

    df["official_store_id"] = df["official_store_id"].astype(str)

    df["catalog_product_id"] = df["catalog_product_id"].astype(str)

    # -- timestamp cols --#
    df["date_created"] = pd.to_datetime(
        df["date_created"], format="%Y-%m-%dT%H:%M:%S.%fZ"
    )
    df["last_updated"] = pd.to_datetime(
        df["last_updated"], format="%Y-%m-%dT%H:%M:%S.%fZ"
    )
    df["days_since_update"] = (df["last_updated"] - df["date_created"]).dt.days
    df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
    df["stop_time"] = pd.to_datetime(df["stop_time"], unit="ms")
    df["days_elapsed"] = (df["stop_time"] - df["start_time"]).dt.days

    ar_holidays = holidays.AR()
    df["wknd_hlday_created"] = df["date_created"].apply(
        is_weekend_or_holiday, args=(ar_holidays,)
    )
    df["wknd_hlday_start_time"] = df["start_time"].apply(
        is_weekend_or_holiday, args=(ar_holidays,)
    )
    df["wknd_hlday_stop_time"] = df["stop_time"].apply(
        is_weekend_or_holiday, args=(ar_holidays,)
    )

    return df


def unnest_dict_columns(df, base_column):
    def unnest_dict(dict_obj, prefix=""):
        # recursive function to unnest the dictionary columns
        unnested_dict = {}
        for k, v in dict_obj.items():
            if isinstance(v, dict):
                unnested_dict.update(unnest_dict(v, prefix=f"{prefix}{k}_"))
            else:
                unnested_dict[f"{prefix}{k}"] = v
        return unnested_dict

    parsed_dict = df[base_column].apply(unnest_dict).apply(pd.Series)

    columns_prefix = parsed_dict.columns.map(lambda x: base_column + str(x))
    parsed_dict = parsed_dict.rename(
        columns=dict(zip(parsed_dict.columns, columns_prefix))
    )

    df = pd.concat([df, parsed_dict], axis=1)

    return df


def extract_pic_features(row):
    if len(row) != 0:
        sizes = [multiply_resolution_str(img["size"]) for img in row]
        num_items = len(row)
        max_size = max(sizes)
        min_size = min(sizes)
        num_sizes = len(set(sizes))
    else:
        num_items = 0
        max_size = None
        min_size = None
        num_sizes = None

    return num_items, max_size, min_size, num_sizes


def multiply_resolution_str(val):
    width, height = map(int, val.split("x"))
    total_value = width * height
    return total_value


def extract_nmp_payment_methods_features(row):
    if len(row) != 0:
        nmp_qty = len(row)
        nmp_cash = any(d.get("description", "").lower() in ["efectivo"] for d in row)
        nmp_giro_postal = any(
            d.get("description", "").lower() in ["giro postal"] for d in row
        )
        nmp_buyer = any(
            d.get("description", "").lower() in ["acordar con el comprador"]
            for d in row
        )
        nmp_mp = any(d.get("description", "").lower() in ["mercadopago"] for d in row)
        nmp_transf = any(
            d.get("description", "").lower() in ["transferencia bancaria"] for d in row
        )
        nmp_tc = any(
            d.get("description", "").lower()
            in [
                "visa electron",
                "mastercard",
                "visa",
                "mastercard",
                "maestro",
                "diners",
                "tarjeta de crédito",
                "american express",
            ]
            for d in row
        )

        return nmp_cash, nmp_giro_postal, nmp_buyer, nmp_mp, nmp_transf, nmp_tc, nmp_qty
    else:
        return None, None, None, None, None, None, None


def parse_descriptions(row):
    if len(row) != 0:
        return re.findall(r"'id': '([^']*)'", row[0])[0]
    else:
        return None


def extract_tag_features(row):
    if len(row):
        good_qt = "good_quality_thumbnail" in row
        poor_qt = "poor_quality_thumbnail" in row
        dragg_v = "dragged_visits" in row
        dragg_bv = "dragged_bids_and_visits" in row
        return good_qt, poor_qt, dragg_v, dragg_bv
    else:
        return None, None, None, None


def unpack_list(lst):
    if lst:
        return lst[0]
    else:
        return None


def get_variations_feature(row):
    try:
        tot_qty = row[0]["sold_quantity"] + row[0]["available_quantity"]
        return tot_qty
    except IndexError:
        return None


def get_warranty_features(x):
    if x is not None:
        no_warranty = bool(re.search(r"\b(sin|sin\s+garantia)\b", x.lower()))
        number_or_yes = bool(re.search(r"\b(\d+\.?\d*|si)\b", x.lower()))
        if no_warranty:
            return False
        elif number_or_yes:
            return True
        else:
            return None


def is_weekend_or_holiday(date, ar_holidays):
    # Check if it's a weekend
    if date.weekday() >= 5:
        return "weekend"
    if date in ar_holidays:
        return "holiday"
    else:
        return "weekday"
