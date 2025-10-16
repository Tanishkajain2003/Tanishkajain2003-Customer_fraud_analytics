import pandas as pd
import numpy as np

def data_quality_checks(df):
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "invalid_amounts": (df['amount'] <= 0).sum()
    }
    return report

def clean_data(df):
    df = df.drop_duplicates()
    df = df[df['amount'] > 0]
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/transactions.csv')
    report = data_quality_checks(df)
    print("Data Quality Report:", report)

    cleaned_df = clean_data(df)
    cleaned_df.to_csv('data/processed/cleaned_transactions.csv', index=False)
    print("Cleaned data saved to data/processed/cleaned_transactions.csv")
