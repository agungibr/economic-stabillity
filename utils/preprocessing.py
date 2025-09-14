import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def create_time_series_features(df):
    df.sort_values(by=['country_code', 'date'], inplace=True)
    
    key_features = [
        'gdp_per_capita', 'inflation_rate', 'unemployment_rate', 
        'interest_rate', 'exchange_rate', 'exports_usd', 'imports_usd',
        'business_confidence_index', 'manufacturing_pmi'
    ]
    
    lags = [1, 7, 14]
    windows = [7, 14, 30]
    
    grouped = df.groupby('country_code')
    
    for col in key_features:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
            
            for window in windows:
                roll_mean = grouped[col].rolling(window, min_periods=1).mean()
                roll_std = grouped[col].rolling(window, min_periods=1).std()
                
                df[f'{col}_roll_mean_{window}'] = roll_mean.reset_index(level=0, drop=True)
                df[f'{col}_roll_std_{window}'] = roll_std.reset_index(level=0, drop=True)
                
    return df

def engineer_date_features(df):
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df.drop(columns=['date'], inplace=True)
    return df

def perform_manual_encoding(df, encoding_maps):
    for column, mapping in encoding_maps.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df

def perform_hybrid_imputation(train_df, test_df):
    impute_cols = [col for col in train_df.columns if pd.api.types.is_numeric_dtype(train_df[col])]
    
    train_df.set_index('date', inplace=True)
    test_df.set_index('date', inplace=True)
    train_df[impute_cols] = train_df[impute_cols].interpolate(method='time', limit_direction='both')
    test_df[impute_cols] = test_df[impute_cols].interpolate(method='time', limit_direction='both')
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)
    
    impute_cols_train = [col for col in impute_cols if col != 'economic_day_status']
    impute_cols_test = [col for col in impute_cols if col != 'id']
    common_impute_cols = list(set(impute_cols_train) & set(impute_cols_test))

    imputer = IterativeImputer(max_iter=10, random_state=42)

    print("Fitting MICE imputer on interpolated data")
    imputer.fit(train_df[common_impute_cols])

    train_df[common_impute_cols] = imputer.transform(train_df[common_impute_cols])
    test_df[common_impute_cols] = imputer.transform(test_df[common_impute_cols])
    return train_df, test_df


def main():
    project_root = Path(__file__).parent.parent
    data_folder = project_root / 'Data'
    res_folder = data_folder / 'Res'
    
    train_file = res_folder / 'train_imputed.csv'
    test_file = res_folder / 'test_imputed.csv'
    
    output_train_file = res_folder / 'train_preprocessed.csv'
    output_test_file = res_folder / 'test_preprocessed.csv'

    encoding_maps = {
        "income_group": {"Lower-Middle": 0, "Upper-Middle": 1},
        "trade_balance_status": {"Deficit": 0, "Neutral": 1, "Surplus": 2},
        "political_stability": {"Unstable": 0, "Moderate": 1, "Stable": 2},
        "economic_sector_dominant": {"Industry": 0, "Services": 1, "Agriculture": 2},
        "currency_type": {"Fiat": 0, "Commodity-backed": 1},
        "policy_framework": {"Socialist": 0, "Keynesian": 1, "Mixed": 2, "Neoliberal": 3},
        "season": {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4},
        "crisis_event": {"InflationShock": 0, "DebtCrisis": 1, "Recession": 2, "None": 3},
        "governance_quality": {"Low": 0, "Medium": 1, "High": 2, "Weak": 0, "Moderate": 1, "Strong": 2},
        "climate_impact_level": {"Low": 0, "Medium": 1, "High": 2},
        "financial_access": {"Low": 0, "Medium": 1, "High": 2},
        "migration_trend": {"Outflow": 0, "Neutral": 1, "Inflow": 2},
        "economic_day_status": {"Low": 0, "Medium": 1, "High": 2}
    }

    print(f"Processing Training Set: {train_file.name}")
    if not train_file.is_file(): return
    train_df = pd.read_csv(train_file, low_memory=False)
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    cols_to_drop = ["region", "trade_bloc", "country_code_file"]
    train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print(f"\nProcessing Testing Set: {test_file.name}")
    if not test_file.is_file(): return
    test_df = pd.read_csv(test_file, low_memory=False)
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = create_time_series_features(combined_df)

    train_df = combined_df[combined_df['date'].dt.year <= 2020].copy()
    test_df = combined_df[combined_df['date'].dt.year > 2020].copy()
    
    print("Processing Training Set:")
    train_df = perform_manual_encoding(train_df, encoding_maps)
    print("\nProcessing Testing Set:")
    test_df = perform_manual_encoding(test_df, encoding_maps)
    
    print("\nPerforming Hybrid Imputation (Interpolate + MICE)")
    train_df, test_df = perform_hybrid_imputation(train_df, test_df)

    print("\nEngineering Date Features")
    train_df = engineer_date_features(train_df)
    test_df = engineer_date_features(test_df)
    
    print("\nPerforming One-Hot Encoding for 'country_code'")
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined_df_final = pd.concat([train_df, test_df], ignore_index=True)
    combined_df_final = pd.get_dummies(combined_df_final, columns=['country_code'], prefix='country', dtype=int)
    
    train_df_final = combined_df_final[combined_df_final['source'] == 'train'].drop(columns=['source'])
    test_df_final = combined_df_final[combined_df_final['source'] == 'test'].drop(columns=['source'])
    
    train_df_final.to_csv(output_train_file, index=False)
    print(f"Saved preprocessed training data to '{output_train_file}'")
    
    test_df_final.to_csv(output_test_file, index=False)
    print(f"Saved preprocessed testing data to '{output_test_file}'")

if __name__ == '__main__':
    main()