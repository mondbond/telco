from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def telco_preprocess(telco_ds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    telco_ds = telco_ds.drop_duplicates()
    telco_ds = telco_ds.drop(columns=["customerID"])
    telco_ds.rename(columns={'gender': 'isMale', 'SeniorCitizen' : 'isSenior', 'Partner':'hasPartner',
                             'Dependents': 'hasDependents', 'PhoneService' : 'hasPhoneService',
                             'PaperlessBilling' : 'isPaperlessBilling'}, inplace=True)

    telco_ds['isMale'] = telco_ds['isMale'].apply(lambda x: 1 if x == 'Male' else 0)
    telco_ds['hasPartner'] = telco_ds['hasPartner'].apply(lambda x: 1 if x == 'Yes' else 0)
    telco_ds['hasDependents'] = telco_ds['hasDependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    telco_ds['hasPhoneService'] = telco_ds['hasPhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    telco_ds['isPaperlessBilling'] = telco_ds['isPaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
    telco_ds['Churn'] = telco_ds['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)


    cat_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    telco_ds = pd.get_dummies(telco_ds, columns=cat_features, drop_first=True)

    bool_cols = telco_ds.select_dtypes('bool').columns
    telco_ds[bool_cols] = telco_ds[bool_cols].fillna(False).astype(int)

    telco_ds['TotalCharges'] = pd.to_numeric(telco_ds['TotalCharges'], errors='coerce')
    telco_ds = telco_ds.dropna(subset=['TotalCharges'])

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()

    # I use standard scaler because TotalCharges has a lot of max outliers
    telco_ds[numeric_features] = scaler.fit_transform(telco_ds[numeric_features])
    y = telco_ds['Churn']
    x = telco_ds.drop(columns='Churn')

    scaler_save_path = (Path(__file__).resolve().parents[4] / "data" / "04_feature" / "scaler.pkl")
    joblib.dump(scaler, scaler_save_path)

    return x, y