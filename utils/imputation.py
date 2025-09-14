import pandas as pd
from pathlib import Path
import sys

def imputation(df_to_impute, data_folder):
    valid_df = df_to_impute[df_to_impute['country_code'].notna()].copy()
    nan_df = df_to_impute[df_to_impute['country_code'].isna()].copy()
    
    unique_countries = valid_df['country_code'].unique()
    imputed_country_dfs = []

    for country in unique_countries:
        print(f"\nProcessing Country: {country}")
        
        country_df = valid_df[valid_df['country_code'] == country].copy()
        
        country_data_path = data_folder / country
        extracted_files = list(country_data_path.rglob('_extracted_data_2000-2024.csv'))
        
        if not extracted_files:
            print(f"No extracted data files found for {country}. Skipping.")
            imputed_country_dfs.append(country_df)
            continue
            
        print(f"Found {len(extracted_files)} extracted files for {country}.")
        
        extracted_dfs_for_country = []
        for file in extracted_files:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            df.set_index('Date', inplace=True)
            extracted_dfs_for_country.append(df)
        
        source_df_for_country = pd.concat(extracted_dfs_for_country, axis=1)
        source_df_for_country = source_df_for_country.T.groupby(level=0).first().T
        source_df_for_country.columns = source_df_for_country.columns.str.lower()
        
        missing_before = country_df.isnull().sum()
        
        country_df.set_index('date', inplace=True)
        country_df.fillna(source_df_for_country, inplace=True)
        country_df.reset_index(inplace=True)
        
        missing_after = country_df.isnull().sum()
        filled_summary = (missing_before - missing_after).loc[lambda x: x > 0]
        
        if not filled_summary.empty:
            print("Imputation summary for this country:")
            for col, count in filled_summary.items():
                if count > 0:
                    print(f"  - Filled {count} missing values in '{col}'.")
        
        imputed_country_dfs.append(country_df)

    imputed_valid_df = pd.concat(imputed_country_dfs, ignore_index=True)
    final_df = pd.concat([imputed_valid_df, nan_df], ignore_index=True)
    
    return final_df

def main():
    project_root = Path(__file__).parent.parent
    data_folder = project_root / 'Data'
    train_file = data_folder / 'train.csv'
    test_file = data_folder / 'test.csv'
    output_folder = data_folder / 'Res'
    output_train_file = output_folder / 'train_imputed.csv'
    output_test_file = output_folder / 'test_imputed.csv'
    output_folder.mkdir(exist_ok=True)

    if train_file.is_file():
        print(f"Processing Training Set ({train_file.name})")
        train_df = pd.read_csv(train_file)
        train_df['date'] = pd.to_datetime(train_df['date']).dt.normalize()
        
        train_df_filtered = train_df[train_df['date'].dt.year <= 2020].copy()
        print(f"Loaded {len(train_df_filtered)} rows for the period 2000-2020.")
        
        imputed_train_df = imputation(train_df_filtered, data_folder)
        
        imputed_train_df.sort_values(by=['country_code', 'date'], inplace=True, na_position='last')
        imputed_train_df.to_csv(output_train_file, index=False)
        print(f"\nSuccessfully saved imputed training data to '{output_train_file}'")
    else:
        print(f"Warning: {train_file.name} not found. Skipping training set.")

    if test_file.is_file():
        print(f"\nProcessing Testing Set ({test_file.name})")
        test_df = pd.read_csv(test_file)
        test_df['date'] = pd.to_datetime(test_df['date']).dt.normalize()
        
        test_df_filtered = test_df[test_df['date'].dt.year >= 2021].copy()
        print(f"Loaded {len(test_df_filtered)} rows for the period 2021-2024.")
        
        imputed_test_df = imputation(test_df_filtered, data_folder)
        
        imputed_test_df.sort_values(by=['country_code', 'date'], inplace=True, na_position='last')
        imputed_test_df.to_csv(output_test_file, index=False)
        print(f"\nSuccessfully saved imputed testing data to '{output_test_file}'")
    else:
        print(f"Warning: {test_file.name} not found. Skipping testing set.")

if __name__ == '__main__':
    main()