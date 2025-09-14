import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from model.LSTM import LSTMModel

def create_test_sequences(full_data, sequence_length, num_test_points):
    xs = []
    for i in range(len(full_data) - sequence_length):
        x = full_data[i:(i + sequence_length)]
        xs.append(x)
    
    return np.array(xs)[-num_test_points:]


def main():
    project_root = Path(__file__).parent
    data_folder = project_root / 'Data'
    res_folder = data_folder / 'Res'
    model_folder = project_root / 'model'
    
    train_file = res_folder / 'train_preprocessed.csv' 
    test_file = res_folder / 'test_preprocessed.csv'
    sample_submission_file = data_folder / 'sample_submission.csv'
    
    model_path = model_folder / 'lstm_model.pt'
    scaler_path = model_folder / 'scaler.pkl'
    submission_file = res_folder / 'agung-2.csv'
    
    SEQUENCE_LENGTH = 30
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 3
    TARGET = 'economic_day_status'
    
    print("--- Loading data, saved model, and scaler ---")
    if not all([train_file.is_file(), test_file.is_file(), model_path.is_file(), scaler_path.is_file(), sample_submission_file.is_file()]):
        print("ERROR: Ensure all required files exist (train/test preprocessed, model, scaler, and sample_submission).")
        return
        
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    scaler = joblib.load(scaler_path)
    
    FEATURES = [col for col in train_df.columns if col not in [TARGET, 'year', 'id']]

    test_df[FEATURES] = scaler.transform(test_df[FEATURES])
    historical_data = train_df.tail(SEQUENCE_LENGTH)[FEATURES]
    full_sequence_data = pd.concat([historical_data, test_df[FEATURES]], ignore_index=True)
    X_test = create_test_sequences(full_sequence_data.values, SEQUENCE_LENGTH, len(test_df))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        test_outputs = model(test_tensor)
        _, final_predictions = torch.max(test_outputs.data, 1)

    submission_df = pd.read_csv(sample_submission_file)
    predictions_numpy = final_predictions.cpu().numpy()
    id_to_prediction_map = dict(zip(test_df['id'], predictions_numpy))
    
    submission_df[TARGET] = submission_df['id'].map(id_to_prediction_map)
    
    if submission_df[TARGET].isnull().any():
        print("Warning: Found IDs in sample_submission that were not in the test data. Filling with 0.")
        submission_df[TARGET].fillna(0, inplace=True)

    submission_df[TARGET] = submission_df[TARGET].astype(int)
    
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\nSuccessfully created submission file: '{submission_file}'")
    print("Submission Head:")
    print(submission_df.head())

if __name__ == '__main__':
    main()