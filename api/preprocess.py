import numpy as np
import pandas as pd 
import pickle


# load onehot encoder
with open("api/preprocessors/onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

# load scaler
with open("api/preprocessors/standard_sc.pkl", "rb") as f:
    scaler = pickle.load(f)


# TODO
COLUMNS_TO_REMOVE = [
    "EmployeeCount", "StandardHours", "Over18"
]

# TODO
COLUMNS_TO_ONEHOT_ENCODE = [
    "BusinessTravel", "Department", "EducationField", 
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]


def preprocess(sample: dict) -> np.ndarray:
    sample_df = pd.DataFrame(sample, index=[0])

    sample_df = drop_columns(sample_df)
    sample_df = encode_columns(sample_df)
    sample_df = create_features(sample_df)
    scaled_sample_values = scale(sample_df.values)
    scaled_sample_values = scaled_sample_values.reshape(1, -1)
    return scaled_sample_values


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_REMOVE)


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    # create a new dataframe with one-hot encoded columns
    encoded_df = pd.DataFrame(onehot_encoder.transform(df[COLUMNS_TO_ONEHOT_ENCODE]).toarray())
    # set new column names
    column_names = onehot_encoder.get_feature_names(COLUMNS_TO_ONEHOT_ENCODE)
    encoded_df.columns = column_names
    # drop raw columns, and add one-hot encoded columns instead
    df = df.drop(columns=COLUMNS_TO_ONEHOT_ENCODE, axis=1)
    df = df.join(encoded_df)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # create MeanAttritionYear feature
    df["MeanAttritionYear"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    df["EmployeeNumberRelationSat"] = df["EmployeeNumber"] / (df["RelationshipSatisfaction"] + 1)
    df["MonthlyIncomeLevel"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)
    df["NumCompaniesJobLevel"] = df["JobLevel"] / (df["NumCompaniesWorked"] + 1)
    df["DailyRateStockOption"] = df["DailyRate"] / (df["StockOptionLevel"] + 1)
    df["MonthlyRateStockOption"] = df["MonthlyRate"] / (df["StockOptionLevel"] + 1)

    # TODO

    return df


def scale(arr: np.ndarray) -> np.ndarray:
    return scaler.transform(arr)
