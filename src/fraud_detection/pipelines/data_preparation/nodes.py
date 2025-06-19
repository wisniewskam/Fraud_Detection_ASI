import pandas as pd

def load_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['Customer_ID', 'Merchant_ID', 'Transaction_ID', 'Customer_Email', 'Customer_Contact', 'Customer_Name', 'Transaction_Currency', 'Bank_Branch'],
        inplace=True, axis=1)
    return df


