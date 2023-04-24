from typing import List, Optional
import pandas as pd
from pandas import DataFrame
import numpy as np


class DataFrameDtypeMapper:
    def __init__(
        self,
        df_columns: list,
        cols_to_bool: list,
        cols_to_float: list,
        cols_to_cat: list,
    ):
        self.column_names_init = df_columns
        self.cols_to_bool = cols_to_bool
        self.cols_to_float = cols_to_float
        self.cols_to_cat = cols_to_cat

    def cast_type(
        self,
        df: DataFrame,
        cast_cat: bool = True,
        cast_bool: bool = True,
        cast_float: bool = True,
    ):
        if cast_cat:
            df[self.cols_to_cat] = df[self.cols_to_cat].astype("category")
        if cast_bool:
            df[self.cols_to_bool] = df[self.cols_to_bool].astype(bool)
        if cast_float:
            df[self.cols_to_float] = df[self.cols_to_float].astype(float)

        return df

    def map_col_names(self, npmatrix: np.ndarray):
        df = pd.DataFrame(npmatrix)
        if len(df.columns) != len(self.column_names_init):
            raise ValueError(
                "Number of columns in col_dict and dataframe are not equal."
            )
        else:
            df.columns = self.column_names_init
            return df
